import os
import time
def mkdir(dir):
    try:
        os.mkdir(dir)
    except OSError:
        pass

def load_image_names_and_path(img_folder_path):
    '''
        Useage: image_names, image_paths, image_names_no_suffix= load_image_names_and_path(img_folder_path)
    :param img_folder_path:
    :return:
    '''
    image_names = next(os.walk(img_folder_path))[2]
    image_paths = []
    image_names_no_suffix = []
    for image_name in image_names:
        image_names_no_suffix.append(os.path.splitext(image_name)[0])
        image_paths.append(img_folder_path + '/' + image_name)
    return image_names, image_paths, image_names_no_suffix

def suitable_unit(value):
    unit_str = 'sec'

    if value > 3600:
        unit_str = 'hour'
        value /= 3600
    elif value < 100:
        pass
    else:
        unit_str = 'min'
        value /= 60

    return unit_str, value

def Display_remain_time(num_done_for_display, num_total, time_start, point_rate=0.10, show_every_epoch=False):
    '''
    Usage format: put in the end of Loop body:
        import time
        from Tianchi_utils import Display_remain_time
        ...
        ...
        # -----------------------------
        # for Display_remain_time
        time_start = time.time()
        num_done_for_display = 0
        # -----------------------------
        loop:
            ...
            ...
            ...
            num_done_for_display = Display_remain_time(num_done_for_display, n_total_image, time_start)
        end of loop

    :param num_done_for_display: num of loop already done
    :param num_total: num of total loop to run
    :param time_start: start time of first loop, after loading all the data if necessary
    :return: num_done_for_display
    '''

    num_done_for_display += 1
    point = int(num_total * point_rate)
    if point == 0:
        point = 1
    time_now = time.time()
    if num_done_for_display % point == 0 or show_every_epoch:
        remain_time = float(time_now - time_start) * (num_total - num_done_for_display) / num_done_for_display
        remain_unit_str, remain_time = suitable_unit(remain_time)
        print(' Already done {:d}/{:d} ({:.2f}%), est remain time: {:.2f} {:s}'.format(
            num_done_for_display, num_total, num_done_for_display / num_total * 100, remain_time, remain_unit_str
        ))
    elif num_done_for_display == 1:  # start successfully
        print(' ----------------------------------------------- ')
        print(' Start Running Successfully! Already done 1/{:d}'.format(num_total))
        print(' The remain time of this task will be automatically estimated and showed every {}%.'.format(point_rate*100))

    elif num_done_for_display == num_total:  # finished
        total_time = float(time_now - time_start)
        aver_time = total_time / num_total

        total_unit_str, total_time = suitable_unit(total_time)
        aver_unit_str, aver_time = suitable_unit(aver_time)

        print(' ----------------------------------------------- ')
        print(' Running Finished!')
        print(' Total time consumption: {:.2f} {:s}, average running time for each loop: {:.4f} {:s}'.format(
            total_time, total_unit_str, aver_time, aver_unit_str
        ))
        print(' ----------------------------------------------- ')
    return num_done_for_display