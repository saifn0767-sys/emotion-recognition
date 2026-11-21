from collections import deque
import hashlib
import os
import pathlib
from pathlib import Path
import re
import threading
import time
import uuid

import cv2
import joblib
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from dashboard.visualizations import show_dashboard
from database.db import EMOTIONS, initialize_db, is_cloud, log_prediction, read_logs
from vision.feature_extraction import (
    EXPECTED_FEATURES,
    extract_all_features,
    load_predictor,
)

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / 'models' / 'ck48_final_model.joblib'
PREDICTOR_FILE = ROOT / 'shape_predictor_68_face_landmarks.dat'


st.set_page_config(
    page_title='Facial Emotion System', page_icon='🙂', layout='wide'
)


def _secret(key):
  try:
    value = st.secrets.get(key)
  except Exception:
    value = None
  return value or os.environ.get(key)


def parse_turn_urls(raw):
  """Accepts a clean comma list or text pasted from the provider

  (e.g. 'urls: "turn:host:80", urls: "turns:host:443?transport=tcp"').
  Keeps only turn:/turns: URLs (STUN is added separately) and removes duplicates.
  """
  found = re.findall(r'turns?:[^\s"\',\]\[]+', str(raw or ''))
  return list(dict.fromkeys(found))


def get_rtc_configuration():
  """STUN always. Adds a TURN relay when TURN_URL / TURN_USERNAME / TURN_CREDENTIAL

  are set (Streamlit secrets or env). Set TURN_FORCE_RELAY = true to test that the
  relay really works (all media is then forced through TURN).
  """
  servers = [{'urls': ['stun:stun.l.google.com:19302']}]
  config = {'iceServers': servers}
  urls = parse_turn_urls(_secret('TURN_URL'))
  user, cred = _secret('TURN_USERNAME'), _secret('TURN_CREDENTIAL')
  if urls and user and cred:
    servers.append(
        {'urls': urls, 'username': str(user), 'credential': str(cred)}
    )
    if str(_secret('TURN_FORCE_RELAY')).lower() in ('1', 'true', 'yes'):
      config['iceTransportPolicy'] = 'relay'
  return config


def turn_status():
  urls = parse_turn_urls(_secret('TURN_URL'))
  if urls and _secret('TURN_USERNAME') and _secret('TURN_CREDENTIAL'):
    return f'TURN relay: configured ({len(urls)} URLs)'
  return 'TURN relay: not configured (STUN only)'


@st.cache_resource
def load_assets():
  if not MODEL_FILE.exists():
    raise FileNotFoundError(f'Model missing: {MODEL_FILE}')
  if not PREDICTOR_FILE.exists():
    raise FileNotFoundError(f'Dlib predictor missing: {PREDICTOR_FILE}')
  model = joblib.load(MODEL_FILE)
  n_in = getattr(model, 'n_features_in_', None)
  if n_in is not None and n_in != EXPECTED_FEATURES:
    raise ValueError(
        f'Model expects {n_in} features but the extractor produces'
        f' {EXPECTED_FEATURES}. Use the model trained by the CK+ script.'
    )
  return model, load_predictor(PREDICTOR_FILE)


initialize_db()


# Shared, thread-safe data exchanged between WebRTC frame callback and Streamlit UI.
class SharedState:

  def __init__(self):
    self.lock = threading.Lock()
    self.latest = {
        'label': None,
        'probabilities': {},
        'confidence': None,
        'error': None,
        'face_found': False,
    }
    self.session_id = uuid.uuid4().hex[:10]
    self.last_logged = 0.0
    self.prediction_count = 0


if 'shared_state' not in st.session_state:
  st.session_state.shared_state = SharedState()
state = st.session_state.shared_state


try:
  model, predictor = load_assets()
except Exception as exc:
  st.error(f'Cannot load model assets: {exc}')
  st.info(
      'Place models/ck48_final_model.joblib and'
      ' shape_predictor_68_face_landmarks.dat in the project root as described'
      ' in README.'
  )
  st.stop()


def predict_from_bytes(data):
  """Decode an uploaded/captured photo and classify it. Returns (annotated_rgb, label, probs) or None."""
  image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
  if image is None:
    return None
  features, face, points = extract_all_features(image, predictor)
  if features is None:
    return image[:, :, ::-1], None, {}
  raw_prob = model.predict_proba(features)[0]
  classes = [str(c) for c in model.classes_]
  probs = {c: float(p) for c, p in zip(classes, raw_prob)}
  label = max(probs, key=probs.get)
  display = image.copy()
  for x, y in points.astype(int):
    cv2.circle(
        display,
        (int(x), int(y)),
        max(1, image.shape[1] // 300),
        (50, 220, 80),
        -1,
    )
  return display[:, :, ::-1], label, probs


class EmotionVideoProcessor(VideoProcessorBase):

  def __init__(self):
    self.last_log_time = 0.0

    # Assets are loaded from Streamlit cached resource in the main thread.
    self.model = model
    self.predictor = predictor
    self.shared = state

    # --- النافذة المنزلقة للتنعيم الزمني (آخر 7 إطارات) ---
    self.window_size = 7
    self.prob_history = deque(maxlen=self.window_size)

  def recv(self, frame):
    image = frame.to_ndarray(format='bgr24')
    display = image.copy()
    try:
      features, face, points = extract_all_features(image, self.predictor)
      if features is None:
        # تفريغ الذاكرة المؤقتة عند اختفاء الوجه لتجنب بقاء نتائج قديمة
        self.prob_history.clear()
        with self.shared.lock:
          self.shared.latest = {
              'label': None,
              'probabilities': {},
              'confidence': None,
              'error': None,
              'face_found': False,
          }
        cv2.putText(
            display,
            'No face detected',
            (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 165, 255),
            2,
        )
      else:
        # حساب الاحتمالات الخام للإطار الحالي
        raw_prob = self.model.predict_proba(features)[0]
        classes = [str(x) for x in self.model.classes_]
        current_probs = {c: float(p) for c, p in zip(classes, raw_prob)}

        # إضافة الاحتمالات الحالية للطابور
        self.prob_history.append(current_probs)

        # --- حساب متوسط الاحتمالات عبر الإطارات الأخيرة (التنعيم) ---
        smoothed_probs = {}
        for c in classes:
          smoothed_probs[c] = float(
              np.mean([hist.get(c, 0.0) for hist in self.prob_history])
          )

        # استخراج التسمية والثقة بناءً على الاحتمالات المنعمة
        label = max(smoothed_probs, key=smoothed_probs.get)
        confidence = float(smoothed_probs[label])

        # رسم المعالم والنص على الإطار
        for x, y in points.astype(int):
          if 0 <= x < display.shape[1] and 0 <= y < display.shape[0]:
            cv2.circle(display, (x, y), 1, (50, 220, 80), -1)

        cv2.putText(
            display,
            f'{label.upper()}  {confidence:.1%}',
            (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (30, 220, 90),
            2,
            cv2.LINE_AA,
        )

        with self.shared.lock:
          self.shared.latest = {
              'label': label,
              'probabilities': smoothed_probs,
              'confidence': confidence,
              'error': None,
              'face_found': True,
          }
          self.shared.prediction_count += 1

        now = time.time()
        # تقييد الكتابة في قاعدة البيانات لمرة واحدة كل ثانيتين
        if now - self.last_log_time >= 2.0:
          log_prediction(self.shared.session_id, label, smoothed_probs)
          self.last_log_time = now

    except Exception as exc:
      with self.shared.lock:
        self.shared.latest = {
            'label': None,
            'probabilities': {},
            'confidence': None,
            'error': str(exc),
            'face_found': False,
        }
      cv2.putText(
          display,
          'Processing error',
          (18, 35),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.55,
          (0, 0, 255),
          2,
      )

    return frame.from_ndarray(display, format='bgr24')


st.title('Facial Emotion Recognition System')
st.caption(
    'CK+ facial-expression classifier • Handcrafted features • Multinomial'
    ' Logistic Regression • No deep learning'
)
page = st.sidebar.radio(
    'Navigation',
    ['Photo Capture', 'Live Recognition', 'Analytics Dashboard', 'Stored Data'],
)
st.sidebar.caption(turn_status())
st.sidebar.caption(
    'Storage: '
    + (
        'cloud SQL database'
        if is_cloud()
        else 'local SQLite (set DATABASE_URL for cloud)'
    )
)

if page == 'Photo Capture':
  st.markdown('### Photo capture')
  st.caption(
      'Take a photo or choose one from your gallery. The image is analysed in'
      ' memory and is not saved; only the predicted result is stored in the'
      ' database.'
  )
  source = st.radio(
      'Source', ['Camera', 'Upload from gallery'], horizontal=True
  )
  shot = (
      st.camera_input('Take a photo')
      if source == 'Camera'
      else st.file_uploader('Choose a photo', type=['jpg', 'jpeg', 'png'])
  )
  if shot is not None:
    data = shot.getvalue()
    result = predict_from_bytes(data)
    if result is None:
      st.error('Could not read this image.')
    else:
      annotated, label, probs = result
      st.image(annotated, use_container_width=True)
      if label is None:
        st.warning(
            'No face detected. Face the camera, use good lighting and try again.'
        )
      else:
        st.metric('Predicted expression', label.upper())
        st.metric('Predicted class probability', f'{probs[label]:.1%}')
        st.bar_chart(
            {
                k: v
                for k, v in sorted(
                    probs.items(), key=lambda x: x[1], reverse=True
                )
            }
        )
        digest = hashlib.md5(data).hexdigest()
        if st.session_state.get('last_photo') != digest:
          log_prediction(f'photo-{state.session_id}', label, probs)
          st.session_state['last_photo'] = digest
        st.caption('Result saved to the SQL database (image not stored).')
  st.caption(
      'Predicted facial-expression labels are not direct measurements of a'
      ' person’s internal emotional state.'
  )

elif page == 'Live Recognition':
  st.markdown('### Live camera')
  col_video, col_result = st.columns([1.65, 1])
  with col_video:
    webrtc_streamer(
        key='emotion-live-camera',
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={
            'video': {
                'width': {'ideal': 640},
                'height': {'ideal': 480},
                'frameRate': {'ideal': 15},
            },
            'audio': False,
        },
        async_processing=True,
        rtc_configuration=get_rtc_configuration(),
    )
  with col_result:
    st.markdown('### Current prediction')
    st_autorefresh(interval=1200, key='live_metrics_refresh')
    with state.lock:
      latest = dict(state.latest)
      probs = dict(latest.get('probabilities') or {})
      count = state.prediction_count
    if latest.get('error'):
      st.error(latest['error'])
    if latest.get('label'):
      st.metric('Predicted expression', latest['label'].upper())
      st.metric("Predicted class probability", f"{latest['confidence']:.1%}")
      st.metric('Frames processed', count)
      if probs:
        st.bar_chart(
            {
                k: v
                for k, v in sorted(
                    probs.items(), key=lambda x: x[1], reverse=True
                )
            }
        )
    else:
      st.info('Start the camera and position your face in view.')
  st.caption(
      'Predicted facial-expression labels are not direct measurements of a'
      ' person’s internal emotional state. Images are not saved by this app.'
  )

elif page == 'Analytics Dashboard':
  st.header('Analytics Dashboard')
  st_autorefresh(interval=5000, key='analytics_refresh')
  show_dashboard(read_logs())

else:
  st.header('Stored data (SQL table: emotion_log)')
  st.dataframe(read_logs(), use_container_width=True)
