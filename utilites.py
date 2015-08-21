def ctr(position):
    """
    Возвращает CTR позиции. Формула получена по таблице значений методом наименьших квадратов.
    Таблица значений утеряна.
    """
    return 34.16200393 / position - 0.83594252


def split_duration(current_conversion_rate, delta, number_of_variations, visitors_per_day, visitors_in_experiment_rate):

    """
    Функция расчитывает минимальное количество дней, необходимое для получения достоверных результатов при
    сплит тестировании. Уровень значимости 0.05, статистическая мощность 0.8
        
    Parameters
    ----------
        current_conversion_rate: numeric, Показатель конверсии на данный момент в долях.
        delta:numeric, минимальное изменение, которое хотим отследит. В долях от текущего показателя конверсий
        number_of_variations: int, Число различных вариаций
        visitors_per_day: numeric, Посетителей в день
        visitors_in_experiment_rate: numeric, Доля посетителей, участвующих в эксперименте

    Returns
    -------
        result: int, количество дней, необходимое для получения достоверных результатов при
                сплит тестировании
    """

    p = current_conversion_rate
    effect_size = p * delta

    per_day_observations = visitors_per_day * visitors_in_experiment_rate
    per_variation_result = 16 * p*(1 - p) / (effect_size**2)
    result = per_variation_result * number_of_variations
    result = round(round(result, 0) / per_day_observations)
    if result == 0:
        print("Less than a day")
    else:
        print(str(result) + " day")
    return result

import pydoc
import core_clusterizetion.algomerative
pydoc.doc(core_clusterizetion.algomerative)