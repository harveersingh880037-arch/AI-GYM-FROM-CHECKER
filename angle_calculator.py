import numpy as np


def calculate_angle(a, b, c):
    """
    Teen points (a, b, c) ke beech ka angle nikalta hai,
    jaha 'b' vertex point hota hai (jaise ki knee ya elbow).
    a, b, c => [x, y] coordinates
    Return: angle in degrees (0-180)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def smooth_value(history_list, new_value, window=5):
    """
    Chote jitter/noise ko hatane ke liye simple moving average.
    history_list: list jisme purane values store hote hai (mutates in place)
    """
    history_list.append(new_value)
    if len(history_list) > window:
        history_list.pop(0)
    return sum(history_list) / len(history_list)
