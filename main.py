import os

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.linear_model import LinearRegression

from interval import Interval


class SolarAnalysis:
    def __init__(self, data_set, files):
        self.data_set = data_set
        self.save_path = None
        self.need_save = True
        self.eps = 1.5 * 1e-4
        self.data_list = []
        self.data_intervals = []
        self.read_data(files)

    def read_data(self, files_with_data):
        for file in files_with_data:
            data_r = genfromtxt(file, delimiter=';', encoding='cp1251')
            self.data_list.append([val[0] for val in data_r][1:201])

            intervals = [Interval([val[0] - self.eps, val[0] + self.eps]) for val in data_r][1:201]
            self.data_intervals.append(intervals)

    def draw_plot(self):
        x = np.arange(1, len(self.data_list[0]) + 1, 1, dtype=int)
        fig, ax = plt.subplots()
        min_y = 1000
        max_y = 0
        for num, data_l in enumerate(self.data_list, start=1):
            ax.plot(x, data_l, label=f'Канал {num}')
            min_y = min(min_y, min(data_l))
            max_y = max(max_y, max(data_l))
        plt.xlabel("n")
        plt.ylabel("mV")
        plt.ylim(min_y - 0.1, max_y + 0.1)
        ax.legend(frameon=False)
        ax.set_title(f'Исходные данные: {self.data_set}')
        if self.need_save:
            plt.savefig(f'{self.save_path}/data.png')
        plt.show()

    def get_out_coef(self, data=None):
        if data is None:
            data = self.data_list

        out_coef = []
        for i in range(len(data[0])):
            if isinstance(data[0][i], Interval):
                val = data[0][i].get_point() / data[1][i].get_point()
            else:
                val = data[0][i] / data[1][i]
            out_coef.append(val)
        return out_coef

    def save_and_show(self, file_name):
        if self.need_save:
            plt.savefig(f'{self.save_path}/{file_name}')
        plt.show()

    def plt_errorbar(self, x, data, yerr, title='', label='', ecolor='g'):
        plt.errorbar(x, data, yerr=yerr, marker='.', linestyle='none', ecolor=ecolor, elinewidth=0.5, capsize=2,
                     capthick=1, label=label)
        plt.xlabel("n")
        plt.ylabel("мВ")
        if title:
            plt.title(title)

    def draw_intervals(self):
        x = np.arange(1, len(self.data_list[0]) + 1, 1, dtype=int)
        for num, data_l in enumerate(self.data_list, start=1):
            print(len(set(data_l)))
            self.plt_errorbar(x, data_l, self.eps, f'Канал {num} с интервалом')
            self.save_and_show(f'interval_chanel_{num}.png')

    def get_linear_regression(self):
        x = np.arange(1, len(self.data_list[0]) + 1, 1, dtype=int)
        x_r = np.array(x).reshape((-1, 1))
        lsm_params = []
        for num, data_l in enumerate(self.data_list, start=1):
            model = LinearRegression().fit(x_r, np.array(data_l))
            k = model.coef_
            b = model.intercept_
            self.plt_errorbar(x, data_l, self.eps, f'Канал {num} регрессия', f'Канал с интервалом')
            plt.plot(x, k * x + b, label=f'Линейная регрессия')
            plt.legend(frameon=False)
            self.save_and_show(f'regression_channel_{num}.png')
            lsm_params.append((k, b))
        return lsm_params

    def get_w_histogram(self, lsm_p_list):
        w_list_arr = []
        for num, data_l in enumerate(self.data_list, start=1):
            w_list = []
            lsm_y = lsm_p_list[num - 1][0] * np.arange(0, len(data_l), 1, dtype=int) + lsm_p_list[num - 1][1]
            nums = []
            for x_num, x in enumerate(data_l, start=0):
                start = x - self.eps
                end = x + self.eps
                y = lsm_y[x_num]
                if start > y:
                    w = (self.eps + (start - y)) / self.eps
                    w_list.append(w)
                    nums.append(x_num)
                elif end < y:
                    w = (self.eps + (y - end)) / self.eps
                    w_list.append(w)
                    nums.append(x_num)
                else:
                    w_list.append(1)
            print(f"Count of weighted: {len(nums)}")
            print(f'Weighted numbers: {nums}')
            w_list_arr.append(w_list)
            plt.hist(w_list, label=str(nums))
            plt.xlabel('Коэф растяжения')
            plt.legend(frameon=False)
            plt.title(f'Гистограмма весов растяжения канал {num}')
            self.save_and_show(f'weight_hist_channel_{num}.png')
        return w_list_arr

    def get_linear(self, w_list, lsm_p_list):
        data_len = len(self.data_list[0])
        x = np.arange(1, data_len + 1, 1, dtype=int)
        yerr = [self.eps] * data_len
        linear_data = []
        for num, data_l in enumerate(self.data_list, start=0):
            for i in range(200):
                yerr[i] *= w_list[num][i]
            data_l = np.array(data_l)
            data_l -= x * lsm_p_list[num][0]
            linear_data.append(data_l)
            self.plt_errorbar(x, data_l, yerr, f'Интервалы после вычитания канал {num + 1}')
            b = [lsm_p_list[num][1]] * data_len
            plt.plot(x, b, label=f'{b[0]}')
            plt.legend(frameon=False)
            self.save_and_show(f'linear_channel_{num + 1}.png')
        return linear_data

    @staticmethod
    def multi_jaccard_metric(interval_list):
        res_inter = interval_list[0]
        res_union = interval_list[0]
        for i in range(1, len(interval_list), 1):
            res_inter = [max(res_inter[0], interval_list[i][0]), min(res_inter[1], interval_list[i][1])]
            res_union = [min(res_union[0], interval_list[i][0]), max(res_union[1], interval_list[i][1])]
        return (res_inter[1] - res_inter[0]) / (res_union[1] - res_union[0])

    def get_inner_r(self, lin_data, R_out, w_list=None):
        step_count = 1000
        r_step = (R_out[1] - R_out[0]) / step_count
        start = R_out[0]
        jaccar_list = []
        while start <= R_out[1]:
            if w_list is not None:
                data_l = [[data_item - self.eps * w_list[0][num], data_item + self.eps * w_list[0][num]]
                          for num, data_item in enumerate(list(lin_data[0]))]
                data_l += [[start * (data_item - self.eps * w_list[1][num]),
                            start * (data_item + self.eps * w_list[1][num])]
                           for num, data_item in enumerate(list(lin_data[1]))]
            else:
                data_l = lin_data[0].copy()
                data_l += [[start * data_item[0], start * data_item[1]] for data_item in lin_data[1]]
            jaccar_list.append((start, self.multi_jaccard_metric(data_l)))
            start += r_step
        plt.plot([R[0] for R in jaccar_list], [R[1] for R in jaccar_list])
        optimal = [(R[0], R[1]) for R in jaccar_list if R[1] >= 0]
        optimal_m = None
        if optimal:
            print(f"Rmin = {optimal[0][0]}")
            print(f"Rmax = {optimal[-1][0]}")
            plt.plot(optimal[0][0], optimal[0][1], 'y*', label=f'Rmin={optimal[0][0]:.5f}')
            plt.plot(optimal[-1][0], optimal[-1][1], 'y*', label=f'Rmax={optimal[-1][0]:.5f}')
            argmaxR = max(optimal, key=lambda opt: opt[1])
            plt.plot(argmaxR[0], argmaxR[1], 'g*', label=f'Ropt={argmaxR[0]:.5f} (JK={argmaxR[1]:.5f})')
            optimal_m = argmaxR
        plt.xlabel("R")
        plt.legend(frameon=False)
        plt.title('Мера Жаккара')
        self.save_and_show('jaccar.png')
        return optimal_m

    def draw_all_intervals(self, lin_data, optimal_m, start=1, w_list=None):
        data_len = (len(lin_data[0]))
        x = np.arange(start, start + data_len, 1, dtype=int)

        if w_list is None:
            y_err = [interval.get_rad() for interval in lin_data[0]]
            y = [interval[1] - y_err[num] for num, interval in enumerate(lin_data[0])]
        else:
            y_err = [self.eps] * data_len
            for i in range(data_len):
                y_err[i] *= w_list[0][i]
            y = lin_data[0]
        plt.errorbar(x, y, yerr=y_err, ecolor='blue', label='Канал 1')

        if w_list is None:
            y_err = [optimal_m * (interval[1] - interval[0]) / 2 for interval in lin_data[1]]
            y = [optimal_m * interval[1] - y_err[num] for num, interval in enumerate(lin_data[1])]
        else:
            y_err = [self.eps] * data_len
            for i in range(data_len):
                y_err[i] *= (w_list[1][i] * optimal_m)
                lin_data[1][i] *= optimal_m
            y = lin_data[1]
        plt.errorbar(x, y, yerr=y_err, ecolor='yellow', label='Канал 2')
        plt.legend(frameon=False)
        plt.title(f'Пересечение интервалов')
        plt.xlabel("n")
        plt.ylabel("мВ")
        self.save_and_show('intersection.png')

    def first_lab(self, save_p):
        self.save_path = save_p
        if not os.path.exists(save_p):
            os.makedirs(save_p)
        self.draw_plot()
        out_coef = self.get_out_coef()
        R_outer = [min(out_coef), max(out_coef)]
        print(f"R = {R_outer}")
        self.draw_intervals()
        lsm_p = self.get_linear_regression()
        print(lsm_p)
        w_l = self.get_w_histogram(lsm_p)
        lin_data = self.get_linear(w_l, lsm_p)
        opt_m = self.get_inner_r(lin_data, R_outer, w_l)
        print(f"Ropt = {opt_m}")
        if opt_m is not None:
            self.draw_all_intervals(lin_data, opt_m[0], w_l)

    def prepare_data_for_matlab(self, output_path):
        with open(f'{output_path}/data_1.mat', "w") as f:
            for i, val in enumerate(self.data_list[0]):
                f.write(f"{i + 1} {val}\n")

        with open(f'{output_path}/data_2.mat', "w") as f:
            for i, val in enumerate(self.data_list[1]):
                f.write(f"{i + 1} {val}\n")

    def get_intervals_regression(self, params, len_of_intervals=200):
        x = np.arange(1, len_of_intervals + 1, 1, dtype=int)
        intervals = []
        for x_ in x:
            interval = Interval([params[0][0] * x_ + params[1][0], params[0][1] * x_ + params[1][1]])
            intervals.append(interval)
        return intervals

    def get_intervals_regression_edge(self, edge_points, params, len_of_intervals=200):
        intervals = []
        for x in range(edge_points[0]):
            intervals.append(Interval([params[0][0][0] * (x + 1) + params[0][1][0], params[0][0][1] * (x + 1) + params[0][1][1]]))
        for x in range(edge_points[0], edge_points[1]):
            intervals.append(Interval([params[1][0][0] * (x + 1) + params[1][1][0], params[1][0][1] * (x + 1) + params[1][1][1]]))
        for x in range(edge_points[1], len_of_intervals):
            intervals.append(Interval([params[2][0][0] * (x + 1) + params[2][1][0], params[2][0][1] * (x + 1) + params[2][1][1]]))
        return intervals

    def draw_interval_regression(self, params, edge_points=None, file_name=''):
        data_len = len(self.data_intervals[0])
        x = np.arange(1, data_len + 1, 1, dtype=int)
        for num, data_l in enumerate(self.data_intervals, start=0):
            intervals = self.data_intervals[num]
            y_err = [interval.get_rad() for interval in intervals]
            y = [interval[1] - y_err[num] for num, interval in enumerate(intervals)]
            self.plt_errorbar(x, y, y_err, label=f'Канал {num + 1}')

            if edge_points is not None:
                reg_intervals = self.get_intervals_regression_edge(edge_points[num], params[num])
            else:
                reg_intervals = self.get_intervals_regression(params=params[num])
            y_err = [interval.get_rad() for interval in reg_intervals]
            y = [interval[1] - y_err[num] for num, interval in enumerate(reg_intervals)]
            title = f'Канал {num + 1} интервальная регрессия' if edge_points is None \
                else f'Канал {num + 1} интервальная регрессия с 3 секциями'
            self.plt_errorbar(x, y, y_err, title, 'Интервальная регрессия',
                              ecolor='k')
            plt.legend(frameon=False)

            self.save_and_show(f"{file_name or 'regression_intervals'}_channel_{num + 1}.png")

    def throw_drift_component(self, drift_params, edge_points=None):
        new_list = []
        for list_num, list_ in enumerate(self.data_intervals):
            new_intervals = []
            if edge_points is None:
                start_enums = [0, 0]
                for num, interval in enumerate(list_, start=start_enums[list_num]):
                    new_interval = Interval([interval[0] - (num + 1) * drift_params[list_num][1],
                                             interval[1] - (num + 1) * drift_params[list_num][0]])
                    new_intervals.append(new_interval)
            else:
                for num, drift_param in enumerate(drift_params[list_num]):
                    if num == 0:
                        new_list_ = list_[:edge_points[list_num][0]]
                        start = 0
                    elif num == 1:
                        new_list_ = list_[edge_points[list_num][0]:edge_points[list_num][1]]
                        start = edge_points[list_num][0]
                    else:
                        new_list_ = list_[edge_points[list_num][1]:]
                        start = edge_points[list_num][1]
                    for num_, interval in enumerate(new_list_, start=start):
                        new_intervals.append(Interval([interval[0] - (num_ + 1) * drift_param[0][1],
                                                       interval[1] - (num_ + 1) * drift_param[0][0]]))
            new_list.append(new_intervals)
        return new_list

    def draw_interval_with_edge(self, edge_points):
        data_len = (len(self.data_intervals[0]))
        x = np.arange(1, data_len + 1, 1, dtype=int)
        for num, data_l in enumerate(self.data_intervals, start=0):
            y_err = [(interval[1] - interval[0]) / 2 for interval in self.data_intervals[num]]
            y = [interval[1] - y_err[num] for num, interval in enumerate(self.data_intervals[num])]
            self.plt_errorbar(x[:edge_points[num][0]], y[:edge_points[num][0]], y_err[:edge_points[num][0]],
                              ecolor='cyan', label=f'set 1, {edge_points[num][0] - 1}')
            self.plt_errorbar(x[edge_points[num][0]:edge_points[num][1]], y[edge_points[num][0]:edge_points[num][1]],
                              y_err[edge_points[num][0]:edge_points[num][1]],
                              ecolor='green', label=f'set {edge_points[num][0]}, {edge_points[num][1] - 1}')
            self.plt_errorbar(x[edge_points[num][1]:], y[edge_points[num][1]:], y_err[edge_points[num][1]:],
                              ecolor='red', label=f'set {edge_points[num][1]}, 200',
                              title=f'Канал {num + 1} участки линейности')

            self.save_and_show(f'sets_of_intervals_channel_{num + 1}.png')

    def process_slice(self, indexes, intervals_regression_params_3, edge_points_):
        out_coeff = self.get_out_coef([self.data_intervals[0][indexes[0]:indexes[1]], self.data_intervals[1][indexes[0]:indexes[1]]])
        R_outer = [min(out_coeff), max(out_coeff)]
        print(R_outer)
        intervals_regression_undrifted = self.throw_drift_component(intervals_regression_params_3, edge_points_)
        opt_m = self.get_inner_r([intervals_regression_undrifted[0][indexes[0]:indexes[1]],
                                  intervals_regression_undrifted[1][indexes[0]:indexes[1]]], R_outer)
        self.draw_all_intervals([intervals_regression_undrifted[0][indexes[0]:indexes[1]],
                                 intervals_regression_undrifted[1][indexes[0]:indexes[1]]], opt_m[0], indexes[0])

    def second_lab(self, save_p):
        self.save_path = save_p
        if not os.path.exists(save_p):
            os.makedirs(save_p)

        # For report channel 1
        # 1.0134e-05
        # 1.1129e-05
        # 1.2234e-05
        # 1.3892e-05
        # 1.5551e-05

        # y_prad channel 2
        # 7.9426e-05
        # 8.3638e-05
        # 8.8317e-05
        # 9.5337e-05
        # 1.0236e-04

        # regression_channel_1 = ([4.0269e-06, 4.2480e-06], [6.6951e-01, 6.6953e-01])
        # regression_channel_2 = ([2.8428e-06, 3.7787e-06], [7.2913e-01, 7.2916e-01])
        # intervals_regression_params = [regression_channel_1, regression_channel_2]
        # self.draw_interval_regression(intervals_regression_params)

        # intervals_regression_drift_params = [regression_channel_1[0], regression_channel_2[0]]
        # intervals_undrifted = self.throw_drift_component(intervals_regression_drift_params)
        #
        # opt_m = self.get_inner_r(intervals_undrifted, [0.9181, 0.9186])
        # self.draw_all_intervals(intervals_undrifted, opt_m[0])

        edge_points_ = [[25, 175], [14, 167]]
        # self.draw_interval_with_edge(edge_points_)

        reg_3_params_ch_1 = [([1.2267e-06, 1.6960e-05], [6.6936e-01, 6.6953e-01]),
                             ([1.7298e-06, 4.2765e-06], [6.6950e-01, 6.6975e-01]),
                             ([3.3957e-06, 1.6960e-05], [6.6710e-01, 6.6965e-01])]

        reg_3_params_ch_2 = [([1.6818e-06, 3.1429e-05], [7.2894e-01, 7.2913e-01]),
                             ([7.3585e-07, 3.4079e-06], [7.2917e-01, 7.2941e-01]),
                             ([6.1333e-07, 1.2114e-05], [7.2754e-01, 7.2963e-01])]
        intervals_regression_params_3 = [reg_3_params_ch_1, reg_3_params_ch_2]

        # self.draw_interval_regression(intervals_regression_params_3, edge_points_, "regression_intervals_edge")

        indexes = [0, max([edge_points_[0][0], edge_points_[1][0]])]
        self.process_slice(indexes, intervals_regression_params_3, edge_points_)

        indexes = [indexes[1], min([edge_points_[0][1], edge_points_[1][1]])]
        self.process_slice(indexes, intervals_regression_params_3, edge_points_)

        indexes = [indexes[1], 200]
        self.process_slice(indexes, intervals_regression_params_3, edge_points_)


if __name__ == "__main__":
    data_set = '700nm_0.23mm.csv'
    files = [f'./data/Канал 1_{data_set}', f'./data/Канал 2_{data_set}']
    analysis = SolarAnalysis(data_set, files)
    # save_p_first = f'./results/first_lab'
    # analysis.first_lab(save_p_first)
    # analysis.prepare_data_for_matlab('./octave/data')
    analysis.second_lab(f'./results/second_lab')
