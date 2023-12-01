import tensorflow_hub as hub
import streamlit as st
import base64
from src.ProganModel import ModelProcessor


@st.cache_resource()
def load_model():
    """загрузка модели. Кешируется Streamlit."""
    model = hub.load(
        "https://kaggle.com/models/google/progan-128/frameworks/tensorflow1/variations/progan-128/versions/1").signatures['default']
    return model


model = load_model()
processor = ModelProcessor(model)


st.header('Модель Progan-128.', divider='orange')
st.subheader('''Модель обучена на датасете фотографий знаменитостей. \
             Генерируется двe фотографии и анимация перехода.''', divider='orange')

is_checked = st.checkbox(
    'Детерминированная последовательность изображений', value=True)


if is_checked and 'seed' not in st.session_state:
    st.session_state.seed = 0


def print_gif(url):
    """Вывод gif через Streamlit."""
    file_ = open(url, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="face gif">',
                unsafe_allow_html=True)


if st.button('Генерировать'):
    seed = None
    if is_checked and 'seed' in st.session_state:
        st.session_state.seed += 1
        seed = st.session_state.seed
    images = processor.interpolate_between_vectors(seed)
    path = processor.animate(images)
    print_gif(path)
