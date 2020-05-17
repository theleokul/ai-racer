import os
import argparse
import gym
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np


# Создаем список путей к моделям в файловой системе
dir_path = os.path.dirname(os.path.abspath(__file__))
model_paths = ['normal.h5', 'cutter.h5', 'drifter.h5']
model_paths = [os.path.join('models', m) for m in model_paths]
model_paths = [os.path.join(dir_path, m) for m in model_paths]


# Добавляем 2 аргумента (номер стиля и числов итераций для пробега) к нашей программе при вызове из командной строки
parser = argparse.ArgumentParser(description='AI racer')
parser.add_argument("-s", "--style", type=int, default=2,
                    help="Driving style: 0 - normal, 1 - cutter, 2 - drifter")
parser.add_argument("-i", "--iterations", type=int, default=3000,
                    help="Frames to proccess, when shut down")
args = parser.parse_args()
# Это номер стиля
driving_style = args.style
# Это число итераций для пробега (число кадров, которые проиграются программой)
iter_count = args.iterations


# Проверяем корректность ввода стиля (поддерживается 3 значения: 0, 1, 2)
if driving_style not in range(len(model_paths)):
    raise Exception('Unsupported driving style, read --help.')
# Загружаем предобученную модель модель
model_path = model_paths[driving_style]
model = load_model(model_path)

# Это небольшой хак, который позволяет машинке ездить лучше
speed_limit = 0.4 if driving_style == 2 else 0.6
break_power = 0.3 if driving_style == 2 else 0.4

# Возможные действия, которые может совершать машинка
ACTIONS = np.array(
    [
        np.array([0, speed_limit, 0]),  # Вперед
        np.array([0, 0, 0]),  # Ничего не делать
        np.array([0, 0, break_power]),  # Тормоз
        np.array([-1, 0, 0]),  # Налево
        np.array([1, 0, 0]),  # Направо
        np.array([-1, 0, break_power]),  # Налево с заносом
        np.array([1, 0, break_power])  # Направо с заносом
    ], dtype=np.float16
)


def main():
    # Это размерность входа модели
    state_shape = (1, 96, 96, 3)
    # Создание модели
    env = gym.make('CarRacing-v0')
    # Получаем начальное состояние
    state = env.reset()
    # Меняем вид состояния к пригодному для входа в модель
    states = state.reshape(state_shape)

    # Начальное ускорение (небольшой хак, всегда можно это сделать в начале любой трассы)
    for _ in range(50):
        # Формируем действие вперед
        action = np.array([0, 1, 0])
        # Подаем его в окружение и получаем обновленное состояние среды
        next_state, reward, done, _ = env.step(action)
        # Отрисовываем изменения среды
        env.render()
        # Меняем вид состояния к пригодному для входа в модель
        states = next_state.reshape(state_shape)

    # Придав начальное ускорение машинке, передаем управление модели
    for _ in range(iter_count):
        # Изменение типа в тип библиотеки tensorflow
        states = tf.cast(states, tf.float16)
        # Делаем предсказание действия
        action_dist = model.predict(states)
        # Выбираем наиболее вероятное
        action_num = np.argmax(action_dist)
        # Достаем описание этого действия из глобальной переменной
        action = ACTIONS[action_num]
        # Эволюционируем окружение
        next_state, reward, done, _ = env.step(action)
        # Отрисовываем изменения
        env.render()
        # Меняем вид состояния к пригодному для входа в модель
        states = next_state.reshape(state_shape)

        # Если дошли до финиша преждевременно, то программа завершается
        if done:
            print('Track successfully completed')
            break

    # Удаляем окружение
    env.close()


if __name__ == "__main__":
    main()
