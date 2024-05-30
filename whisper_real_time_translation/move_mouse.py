import pyautogui


def resize():
    mouse = pyautogui
    monitor_size = mouse.size()
    time_1_rel_x, time_1_rel_y = 0.72421875, 0.9618055
    time_2_rel_x, time_2_rel_y = 0.5734375, 0.9618055
    time_3_rel_x, time_3_rel_y = 0.57109375, 0.9277778

    time_1_x, time_1_y = time_1_rel_x*monitor_size[0], time_1_rel_y*monitor_size[1]
    time_2_x, time_2_y = time_2_rel_x*monitor_size[0], time_2_rel_y*monitor_size[1]
    time_3_x, time_3_y = time_3_rel_x*monitor_size[0], time_3_rel_y*monitor_size[1]

    time_1 = (time_1_x, time_1_y)
    time_2 = (time_2_x, time_2_y)
    time_3 = (time_3_x, time_3_y)

    return time_1, time_2, time_3

def handraiser():
    mouse = pyautogui
    original = pyautogui.position()
    time_1, time_2, time_3 = resize()
    mouse.sleep(0.1)
    mouse.moveTo(time_1[0], time_1[0])
    mouse.sleep(0.5)
    mouse.click(time_2[0],time_2[1])
    mouse.sleep(0.05)
    mouse.click(time_3[0],time_3[1])
    mouse.moveTo(original[0],original[0])