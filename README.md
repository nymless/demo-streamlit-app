# Demo Streamlit App

**Для копирования проекта и перехода в рабочюю директорию необходимо выполнить:**

    git clone https://github.com/nymless/demo-streamlit-app.git
    cd demo-streamlit-app

**Установка окружения и зависимостей:**

    pip install virtualenv
    virtualenv venv
    source venv/Script/activate
    python -m pip install -r requirements.txt

**Без requirements.txt зависимости следующие:**

    python -m pip install numpy tensorflow tensorflow_hub imageio ipython streamlit

**Запуск приложения в Streamlit:**

    streamlit run main.py

При нажатии на кнопку "Генерация" происходит генерация двух фотографий и анимация перехода между ними.

По умолчанию включен режим, при котором очерёдность генерации фотографий детерминирована, т.е. каждый раз одинакова.

Использована модель Progan-128, предобученная на датасете фотографий знаменитостей.
