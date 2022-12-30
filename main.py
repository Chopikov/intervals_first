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
        self.edge_points = None
        self.intervals_regression_params_3 = None
        self.intervals_regression_params = None

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

    def process_slice(self, indexes, intervals_regression_params_3, edge_points):
        out_coeff = self.get_out_coef([self.data_intervals[0][indexes[0]:indexes[1]], self.data_intervals[1][indexes[0]:indexes[1]]])
        R_outer = [min(out_coeff), max(out_coeff)]
        print(R_outer)
        intervals_regression_undrifted = self.throw_drift_component(intervals_regression_params_3, edge_points)
        opt_m = self.get_inner_r([intervals_regression_undrifted[0][indexes[0]:indexes[1]],
                                  intervals_regression_undrifted[1][indexes[0]:indexes[1]]], R_outer)
        self.draw_all_intervals([intervals_regression_undrifted[0][indexes[0]:indexes[1]],
                                 intervals_regression_undrifted[1][indexes[0]:indexes[1]]], opt_m[0], indexes[0])

    def fill_params(self):
        regression_channel_1 = ([4.0269e-06, 4.2480e-06], [6.6951e-01, 6.6953e-01])
        regression_channel_2 = ([2.8428e-06, 3.7787e-06], [7.2913e-01, 7.2916e-01])
        self.intervals_regression_params = [regression_channel_1, regression_channel_2]

        reg_3_params_ch_1 = [([1.2267e-06, 1.6960e-05], [6.6936e-01, 6.6953e-01]),
                             ([1.7298e-06, 4.2765e-06], [6.6950e-01, 6.6975e-01]),
                             ([3.3957e-06, 1.6960e-05], [6.6710e-01, 6.6965e-01])]

        reg_3_params_ch_2 = [([1.6818e-06, 3.1429e-05], [7.2894e-01, 7.2913e-01]),
                             ([7.3585e-07, 3.4079e-06], [7.2917e-01, 7.2941e-01]),
                             ([6.1333e-07, 1.2114e-05], [7.2754e-01, 7.2963e-01])]
        intervals_regression_params_3 = [reg_3_params_ch_1, reg_3_params_ch_2]
        self.intervals_regression_params_3 = intervals_regression_params_3

        edge_points = [[25, 175], [14, 167]]
        self.edge_points = edge_points

    def second_lab(self, save_p):
        self.save_path = save_p
        if not os.path.exists(save_p):
            os.makedirs(save_p)

        self.fill_params()

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

        # self.draw_interval_regression(self.intervals_regression_params)

        # intervals_regression_drift_params = [regression_channel_1[0], regression_channel_2[0]]
        # intervals_undrifted = self.throw_drift_component(intervals_regression_drift_params)
        #
        # opt_m = self.get_inner_r(intervals_undrifted, [0.9181, 0.9186])
        # self.draw_all_intervals(intervals_undrifted, opt_m[0])

        # self.draw_interval_with_edge(edge_points)

        # self.draw_interval_regression(self.intervals_regression_params_3, self.edge_points, "regression_intervals_edge")

        indexes = [0, max([self.edge_points[0][0], self.edge_points[1][0]])]
        self.process_slice(indexes, self.intervals_regression_params_3, self.edge_points)

        indexes = [indexes[1], min([self.edge_points[0][1], self.edge_points[1][1]])]
        self.process_slice(indexes, self.intervals_regression_params_3, self.edge_points)

        indexes = [indexes[1], 200]
        self.process_slice(indexes, self.intervals_regression_params_3, self.edge_points)

    def draw_residuals(self, residuals, intersection_, title='', channel_num=1):
        data_len = (len(residuals))
        x = np.arange(1, data_len + 1, 1, dtype=int)
        y_err = [interval.get_rad() for interval in residuals]
        y = [interval.get_mid() for interval in residuals]
        plt.errorbar(x, y, yerr=y_err, ecolor='yellow', label=f'остатки', elinewidth=0.8, capsize=4, capthick=1)
        x1, y1 = [1, 200], [intersection_[0], intersection_[0]]
        x2, y2 = [1, 200], [intersection_[1], intersection_[1]]
        x3, y3 = [1, 200], [(intersection_[1] + intersection_[0]) / 2, (intersection_[1] + intersection_[0]) / 2]
        plt.plot(x1, y1, 'b--', label=f'[{y1[0]:.2e}, {y2[1]:.2e}]')
        plt.plot(x2, y2, 'b--')
        plt.plot(x3, y3, 'k--', label=f'mid={y3[0]:.2e}')
        plt.legend(frameon=False)
        plt.title(title)
        plt.yticks(fontsize=5)
        plt.xlabel("n")
        plt.ylabel("Остаток")
        if self.save_path is not None:
            plt.savefig(f'{self.save_path}/residuals_central_section_ch_{channel_num}.png')
        plt.show()

    def get_intersections(self, interval_list):
        res = interval_list[0]
        for i in range(1, len(interval_list), 1):
            res = Interval([max(min(res), min(interval_list[i])), min(max(res), max(interval_list[i]))])
        return res

    def get_intersections_wrong_int(self, interval_list):
        res = interval_list[0]
        for i in range(1, len(interval_list), 1):
            res = Interval([max(res[0], interval_list[i][0]), min(res[1], interval_list[i][1])])
        return res

    def get_influences(self, interval_list, intersection_=None):
        if intersection_ is not None:
            intersection = intersection_
        else:
            intersection = self.get_intersections(interval_list)
        inter_rad = intersection.get_rad()
        inter_mid = intersection.get_mid()
        influences = []
        for interval in interval_list:
            l = inter_rad / interval.get_rad()
            r = (interval.get_mid() - inter_mid) / interval.get_rad()
            influences.append([l, r])
        return influences, intersection

    def get_residuals(self, interval_d, edge_points, drift_params_3):
        res = []
        for list_num, list_ in enumerate(interval_d):
            interval_list = []
            for num, drift_param in enumerate(drift_params_3[list_num]):
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
                    interval_list.append(Interval([interval[0] - (num_ + 1) * drift_param[0][1] - drift_param[1][1],
                                                   interval[1] - (num_ + 1) * drift_param[0][0] - drift_param[1][0]]))
            res.append(interval_list)
        return res

    def get_residuals_1(self, interval_d, drift_params):
        res = []
        for list_num, list_ in enumerate(interval_d):
            interval_list = []
            for num, drift_param in enumerate(drift_params[list_num]):
                for num_, interval in enumerate(list_, start=0):
                    interval_list.append(Interval([interval[0] - (num_ + 1) * drift_param[0][1] - drift_param[1][1],
                                                   interval[1] - (num_ + 1) * drift_param[0][0] - drift_param[1][0]]))
            res.append(interval_list)
        return res

    def add_point(self, point, ax):
        ax.plot(point[0], point[1], 'bo')

    def draw_data_status_template(self, x_lims=(0, 2), title='Influences'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.patch.set_facecolor('yellow')
        ax.set_xlim(x_lims[0], x_lims[1])
        ax.set_ylim(-(x_lims[1] + 1), x_lims[1] + 1)
        # draw green triangle zone
        x1, y1 = [0, 1], [-1, 0]
        x2, y2 = [0, 1], [1, 0]
        ax.plot(x1, y1, 'k', x2, y2, 'k')
        ax.fill_between(x1, y1, y2, facecolor='green')

        # draw others zones
        x1, y1 = [0, x_lims[1]], [-1, -(x_lims[1] + 1)]
        x2, y2 = [0, x_lims[1]], [1, x_lims[1] + 1]

        ax.plot(x1, y1, 'k', x2, y2, 'k')
        x = np.arange(0.0, x_lims[1], 0.01)
        y1 = x + 1
        y2 = [x_lims[1] + 1] * len(x)
        ax.fill_between(x, y1, y2, facecolor='red')
        y2 = [-(x_lims[1] + 1)] * len(x)
        ax.fill_between(x, -y1, y2, facecolor='red')

        x1, y1 = [1, 1], [-(x_lims[1] + 1), x_lims[1] + 1]
        ax.plot(x1, y1, 'k--')
        ax.set_xlabel('l(x, y)')
        ax.set_ylabel('r(x, y)')
        ax.set_title(title)
        return fig, ax

    def draw_data_status_with_points(self, intervals_residuals, title_redisuals, title_influences):
        for num_, res_list in enumerate(intervals_residuals):
            point = self.edge_points[num_]
            residuals = intervals_residuals[num_][point[0]:point[1]]
            inter_w = self.get_intersections_wrong_int(residuals)
            print(inter_w.points)
            inters = self.get_intersections(residuals)
            infls, intersection = self.get_influences(res_list, inters)
            self.draw_residuals(res_list, intersection, title=f'{title_redisuals}, канал {num_ + 1}')
            m_l = max([res[0] for res in infls])
            fig_, ax_ = self.draw_data_status_template([0, max(m_l, 2)],
                                                       title=f'{title_influences}, канал {num_ + 1}')
            for infl in infls:
                self.add_point(infl, ax_)
            fig_.show()

    def regularization(self, residuals, edge_point, intersection_, need_visualize=False, title='', label=''):
        w_list = [1] * len(residuals)
        left_intervals = residuals[:edge_point].copy()
        for num, interval in enumerate(left_intervals):
            mid = left_intervals[num].get_mid()
            if interval[0] > intersection_[0]:
                left_intervals[num][0] = intersection_[0]
                left_intervals[num][1] = mid + (mid - intersection_[0])
                w_list[num] = (mid - intersection_[0]) / interval.get_rad()
            if interval[1] < intersection_[1]:
                left_intervals[num][1] = intersection_[1]
                left_intervals[num][0] = mid - (intersection_[1] - mid)
                w_list[num] = (intersection_[1] - mid) / interval.get_rad()
        if need_visualize:
            plt.hist(w_list, label=label)
            plt.xlabel(label)
            plt.legend(frameon=False)
            plt.title(title)
            plt.show()
        return left_intervals + residuals[edge_point:]

    def draw_params(self, params_1, params_2=None, title='', param_name='', param_name_2=''):
        x = np.arange(1, len(params_1) + 1, 1, dtype=int)
        fig, ax = plt.subplots()
        if params_2:
            ax.plot(x, np.fabs(np.array(params_1)), label=param_name)
            ax.plot(x, 1 - np.array(params_2), label=param_name_2)
        else:
            ax.plot(x, params_1, label=param_name)
        plt.xlabel("n")
        plt.ylabel(param_name)
        ax.legend()
        ax.set_title(title)
        plt.show()

    def draw_mode(self, mode_data, title='Mode'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data_len = len(mode_data)
        for num, mode_data_inst in enumerate(mode_data):
            x1, y1 = [mode_data_inst[0][0], mode_data_inst[0][1]], [mode_data_inst[1], mode_data_inst[1]]
            ax.plot(x1, y1, 'b')
            if num < data_len - 1:
                x2, y2 = [mode_data_inst[0][1], mode_data_inst[0][1]], [mode_data_inst[1], mode_data[num + 1][1]]
                ax.plot(x2, y2, 'b')
        ax.set_xlabel('T')
        ax.set_ylabel('частота')
        ax.set_title(title)
        plt.show()

    def get_octave_intervals(self, intervals):
        res_str = '['
        for interval in intervals:
            res_str = f'{res_str}[{interval[0]}, {interval[1]}];'
        print(f'{res_str[:-1]}]')

    def third_lab(self, save_p):
        self.save_path = save_p
        if not os.path.exists(save_p):
            os.makedirs(save_p)

        self.fill_params()

        # self.get_octave_intervals(self.data_intervals[1])
        # from_octave_1 = [([0.66927, 0.66933], 1), ([0.66933, 0.66939], 3), ([0.66939, 0.66945], 12), ([0.66945, 0.66951], 15), ([0.66951, 0.66957], 24), ([0.66957, 0.66957], 46), ([0.66957, 0.66963], 45), ([0.66963, 0.66963], 59), ([0.66963, 0.66969], 57), ([0.66969, 0.66969], 80), ([0.66969, 0.66975], 71), ([0.66975, 0.66975], 84), ([0.66975, 0.66981], 81), ([0.66981, 0.66981], 106), ([0.66981, 0.66987], 97), ([0.66987, 0.66987], 118), ([0.66987, 0.66993], 96), ([0.66993, 0.66993], 113), ([0.66993, 0.66999], 99), ([0.66999, 0.66999], 115), ([0.66999, 0.67005], 92), ([0.67005, 0.67005], 97), ([0.67005, 0.67011], 84), ([0.67011, 0.67011], 95), ([0.67011, 0.67017], 70), ([0.67017, 0.67017], 75), ([0.67017, 0.67023], 54), ([0.67023, 0.67023], 55), ([0.67023, 0.67029], 38), ([0.67029, 0.67029], 41), ([0.67029, 0.67035], 25), ([0.67035, 0.67041], 20), ([0.67041, 0.67047], 9), ([0.67047, 0.67053], 4), ([0.67053, 0.67059], 3)]
        # self.draw_mode(from_octave_1, title='Мода для канала 1')
        # from_octave_2 = [([0.72888, 0.72894], 2), ([0.72894, 0.729], 3), ([0.729, 0.72906], 4), ([0.72906, 0.72912], 12), ([0.72912, 0.72918], 13), ([0.72918, 0.72918], 35), ([0.72918, 0.72924], 33), ([0.72924, 0.72924], 61), ([0.72924, 0.7293], 60), ([0.7293, 0.7293], 96), ([0.7293, 0.72935], 95), ([0.72935, 0.72936], 125), ([0.72936, 0.72941], 117), ([0.72941, 0.72942], 128), ([0.72942, 0.72947], 127), ([0.72947, 0.72948], 154), ([0.72948, 0.72953], 132), ([0.72953, 0.72954], 147), ([0.72954, 0.72959], 119), ([0.72959, 0.7296], 128), ([0.7296, 0.72965], 92), ([0.72965, 0.72965], 97), ([0.72965, 0.72971], 67), ([0.72971, 0.72971], 71), ([0.72971, 0.72977], 60), ([0.72977, 0.72983], 33), ([0.72983, 0.72989], 18), ([0.72989, 0.72995], 9), ([0.72995, 0.73001], 4)]
        # self.draw_mode(from_octave_2, title='Мода для канала 2')

        #
        # intervals_residuals_1 = self.get_residuals_1(self.data_intervals, [[self.intervals_regression_params_3[0][1]],
        #                                                                    [self.intervals_regression_params_3[1][1]]])
        # self.get_octave_intervals(intervals_residuals_1[0])
        # self.draw_data_status_with_points(intervals_residuals_1, 'Остатки для центральной подвыборки',
        #                                   'Диаграмма статусов для центральной подвыборки')
        # from_octave_residuals_1 = [([-0.00050969, -0.00050541], 1), ([-0.00050541, -0.00050113], 2), ([-0.00050113, -0.00050086], 3), ([-0.00050086, -0.00049686], 4), ([-0.00049686, -0.00049659], 5), ([-0.00049659, -0.00049258], 6), ([-0.00049258, -0.00049231], 7), ([-0.00049231, -0.0004883], 8), ([-0.0004883, -0.00048803], 9), ([-0.00048803, -0.00048776], 10), ([-0.00048776, -0.00048403], 11), ([-0.00048403, -0.00048376], 12), ([-0.00048376, -0.00048349], 13), ([-0.00048349, -0.00048078], 14), ([-0.00048078, -0.00047975], 15), ([-0.00047975, -0.00047948], 16), ([-0.00047948, -0.00047921], 17), ([-0.00047921, -0.00047548], 18), ([-0.00047548, -0.0004752], 19), ([-0.0004752, -0.00047493], 20), ([-0.00047493, -0.00047137], 21), ([-0.00047137, -0.0004712], 22), ([-0.0004712, -0.00047093], 23), ([-0.00047093, -0.00047066], 24), ([-0.00047066, -0.00046709], 25), ([-0.00046709, -0.00046692], 26), ([-0.00046692, -0.00046665], 27), ([-0.00046665, -0.00046638], 28), ([-0.00046638, -0.00046282], 29), ([-0.00046282, -0.00046265], 30), ([-0.00046265, -0.00046237], 31), ([-0.00046237, -0.0004621], 32), ([-0.0004621, -0.00045881], 33), ([-0.00045881, -0.00045854], 34), ([-0.00045854, -0.00045837], 35), ([-0.00045837, -0.0004581], 36), ([-0.0004581, -0.00045783], 37), ([-0.00045783, -0.00045756], 38), ([-0.00045756, -0.00045453], 39), ([-0.00045453, -0.00045426], 40), ([-0.00045426, -0.00045409], 41), ([-0.00045409, -0.00045382], 42), ([-0.00045382, -0.00045355], 43), ([-0.00045355, -0.00045328], 44), ([-0.00045328, -0.00045026], 45), ([-0.00045026, -0.00044982], 46), ([-0.00044982, -0.00044955], 47), ([-0.00044955, -0.00044927], 48), ([-0.00044927, -0.000449], 49), ([-0.000449, -0.00044598], 50), ([-0.00044598, -0.00044554], 51), ([-0.00044554, -0.00044527], 52), ([-0.00044527, -0.000445], 53), ([-0.000445, -0.00044473], 54), ([-0.00044473, -0.00044171], 55), ([-0.00044171, -0.00044099], 56), ([-0.00044099, -0.00044072], 57), ([-0.00044072, -0.00044045], 58), ([-0.00044045, -0.00043743], 59), ([-0.00043743, -0.00043672], 60), ([-0.00043672, -0.00043644], 61), ([-0.00043644, -0.00043617], 62), ([-0.00043617, -0.00043315], 63), ([-0.00043315, -0.00043244], 64), ([-0.00043244, -0.00043217], 65), ([-0.00043217, -0.0004319], 66), ([-0.0004319, -0.00042973], 67), ([-0.00042973, -0.00042888], 68), ([-0.00042888, -0.00042789], 69), ([-0.00042789, -0.00042762], 70), ([-0.00042762, -0.00042545], 71), ([-0.00042545, -0.0004246], 72), ([-0.0004246, -0.00042362], 73), ([-0.00042362, -0.00042334], 74), ([-0.00042334, -0.00042059], 75), ([-0.00042059, -0.00042032], 76), ([-0.00042032, -0.00041934], 77), ([-0.00041934, -0.00041907], 78), ([-0.00041907, -0.00041632], 79), ([-0.00041632, -0.00041605], 80), ([-0.00041605, -0.00041506], 81), ([-0.00041506, -0.00041479], 82), ([-0.00041479, -0.00041425], 83), ([-0.00041425, -0.00041204], 84), ([-0.00041204, -0.00041079], 85), ([-0.00041079, -0.00041051], 86), ([-0.00041051, -0.00041024], 87), ([-0.00041024, -0.00040997], 88), ([-0.00040997, -0.00040862], 89), ([-0.00040862, -0.00040776], 90), ([-0.00040776, -0.00040651], 91), ([-0.00040651, -0.00040624], 92), ([-0.00040624, -0.00040597], 93), ([-0.00040597, -0.0004057], 94), ([-0.0004057, -0.00040434], 95), ([-0.00040434, -0.00040349], 96), ([-0.00040349, -0.00040223], 97), ([-0.00040223, -0.00040196], 98), ([-0.00040196, -0.00040169], 99), ([-0.00040169, -0.00040142], 100), ([-0.00040142, -0.00040006], 101), ([-0.00040006, -0.00039769], 102), ([-0.00039769, -0.00039741], 103), ([-0.00039741, -0.00039714], 104), ([-0.00039714, -0.00039579], 105), ([-0.00039579, -0.00039341], 106), ([-0.00039341, -0.00039314], 107), ([-0.00039314, -0.00039287], 108), ([-0.00039287, -0.00039151], 109), ([-0.00039151, -0.00038913], 110), ([-0.00038913, -0.00038886], 111), ([-0.00038886, -0.00038859], 112), ([-0.00038859, -0.00038724], 113), ([-0.00038724, -0.00038486], 114), ([-0.00038486, -0.00038458], 115), ([-0.00038458, -0.00038431], 116), ([-0.00038431, -0.00038296], 117), ([-0.00038296, -0.00038058], 118), ([-0.00038058, -0.00038031], 119), ([-0.00038031, -0.00038004], 120), ([-0.00038004, -0.00037868], 121), ([-0.00037868, -0.0003763], 122), ([-0.0003763, -0.00037603], 123), ([-0.00037603, -0.00037576], 124), ([-0.00037576, -0.00037549], 125), ([-0.00037549, -0.00037522], 126), ([-0.00037522, -0.00037441], 127), ([-0.00037441, -0.00037203], 128), ([-0.00037203, -0.00037176], 129), ([-0.00037176, -0.00037148], 130), ([-0.00037148, -0.00037121], 131), ([-0.00037121, -0.00037094], 132), ([-0.00037094, -0.00036775], 133), ([-0.00036775, -0.00036748], 134), ([-0.00036748, -0.00036721], 135), ([-0.00036721, -0.00036694], 136), ([-0.00036694, -0.00036667], 137), ([-0.00036667, -0.00036527], 138), ([-0.00036527, -0.00036347], 139), ([-0.00036347, -0.0003632], 140), ([-0.0003632, -0.00036293], 141), ([-0.00036293, -0.00036266], 142), ([-0.00036266, -0.00036239], 143), ([-0.00036239, -0.00036185], 144), ([-0.00036185, -0.0003592], 145), ([-0.0003592, -0.00035893], 146), ([-0.00035893, -0.00035865], 147), ([-0.00035865, -0.00035838], 148), ([-0.00035838, -0.00035811], 149), ([-0.00035811, -0.00035757], 150), ([-0.00035757, -0.00035492], 151), ([-0.00035492, -0.00035438], 152), ([-0.00035438, -0.00035411], 153), ([-0.00035411, -0.00035384], 154), ([-0.00035384, -0.00035329], 155), ([-0.00035329, -0.0003501], 156), ([-0.0003501, -0.00034983], 157), ([-0.00034983, -0.00034956], 158), ([-0.00034956, -0.00034583], 159), ([-0.00034583, -0.00034555], 160), ([-0.00034555, -0.00034528], 161), ([-0.00034528, -0.00034155], 162), ([-0.00034155, -0.00034128], 163), ([-0.00034128, -0.00034101], 164), ([-0.00034101, -0.00034074], 165), ([-0.00034074, -0.00033727], 166), ([-0.00033727, -0.000337], 167), ([-0.000337, -0.00033673], 168), ([-0.00033673, -0.00033646], 169), ([-0.00033646, -0.000333], 170), ([-0.000333, -0.00033272], 171), ([-0.00033272, -0.00033245], 172), ([-0.00033245, -0.00033218], 173), ([-0.00033218, -0.00032872], 174), ([-0.00032872, -0.00032845], 175), ([-0.00032845, -0.00032818], 176), ([-0.00032818, -0.00032791], 177), ([-0.00032791, -0.00032444], 178), ([-0.00032444, -0.00032417], 179), ([-0.00032417, -0.0003239], 180), ([-0.0003239, -0.00032363], 181), ([-0.00032363, -0.00032017], 182), ([-0.00032017, -0.0003199], 183), ([-0.0003199, -0.00031962], 184), ([-0.00031962, -0.00031935], 185), ([-0.00031935, -0.0003185], 186), ([-0.0003185, -0.00031535], 187), ([-0.00031535, -0.00031508], 188), ([-0.00031508, -0.00031422], 189), ([-0.00031422, -0.00031107], 190), ([-0.00031107, -0.0003108], 191), ([-0.0003108, -0.00030995], 192), ([-0.00030995, -0.00030679], 193), ([-0.00030679, -0.00030652], 194), ([-0.00030652, -0.00030252], 195), ([-0.00030252, -0.00029824], 196), ([-0.00029824, -0.00029397], 197), ([-0.00029397, -0.00028969], 198), ([-0.00028969, -0.00028541], 199), ([-0.00028541, 7.177e-05], 200), ([7.177e-05, 0.00012791], 199), ([0.00012791, 0.00012964], 198), ([0.00012964, 0.00017194], 197), ([0.00017194, 0.00017367], 196), ([0.00017367, 0.0001754], 195), ([0.0001754, 0.00017713], 194), ([0.00017713, 0.00017886], 193), ([0.00017886, 0.00018059], 192), ([0.00018059, 0.00018232], 191), ([0.00018232, 0.00018405], 190), ([0.00018405, 0.00018578], 189), ([0.00018578, 0.00022635], 188), ([0.00022635, 0.00022808], 187), ([0.00022808, 0.00022981], 186), ([0.00022981, 0.00027038], 185), ([0.00027038, 0.00027211], 184), ([0.00027211, 0.00027384], 183), ([0.00027384, 0.00027557], 182), ([0.00027557, 0.0002773], 181), ([0.0002773, 0.00027903], 180), ([0.00027903, 0.00028076], 179), ([0.00028076, 0.00028249], 178), ([0.00028249, 0.00028422], 177), ([0.00028422, 0.00029193], 176), ([0.00029193, 0.00029366], 175), ([0.00029366, 0.00029539], 174), ([0.00029539, 0.00029712], 173), ([0.00029712, 0.00029885], 172), ([0.00029885, 0.00030058], 171), ([0.00030058, 0.00030231], 170), ([0.00030231, 0.00030404], 169), ([0.00030404, 0.00030577], 168), ([0.00030577, 0.0003075], 167), ([0.0003075, 0.00030923], 166), ([0.00030923, 0.00031096], 165), ([0.00031096, 0.00031269], 164), ([0.00031269, 0.00031442], 163), ([0.00031442, 0.00031615], 162), ([0.00031615, 0.00031788], 161), ([0.00031788, 0.00031961], 160), ([0.00031961, 0.00032134], 159), ([0.00032134, 0.00032307], 158), ([0.00032307, 0.0003248], 157), ([0.0003248, 0.00032653], 156), ([0.00032653, 0.00032731], 155), ([0.00032731, 0.00032825], 154), ([0.00032825, 0.00032904], 153), ([0.00032904, 0.00033077], 152), ([0.00033077, 0.0003325], 151), ([0.0003325, 0.00033423], 150), ([0.00033423, 0.00033596], 149), ([0.00033596, 0.00033769], 148), ([0.00033769, 0.00033942], 147), ([0.00033942, 0.00034115], 146), ([0.00034115, 0.00034288], 145), ([0.00034288, 0.00034461], 144), ([0.00034461, 0.00034634], 143), ([0.00034634, 0.00034713], 142), ([0.00034713, 0.00034807], 141), ([0.00034807, 0.00034886], 140), ([0.00034886, 0.0003498], 139), ([0.0003498, 0.00035059], 138), ([0.00035059, 0.00035232], 137), ([0.00035232, 0.00035405], 136), ([0.00035405, 0.00035578], 135), ([0.00035578, 0.00035751], 134), ([0.00035751, 0.00035924], 133), ([0.00035924, 0.00036096], 132), ([0.00036096, 0.00036269], 131), ([0.00036269, 0.00036442], 130), ([0.00036442, 0.00036615], 129), ([0.00036615, 0.00036788], 128), ([0.00036788, 0.00036961], 127), ([0.00036961, 0.00037134], 126), ([0.00037134, 0.00037307], 125), ([0.00037307, 0.0003748], 124), ([0.0003748, 0.00037653], 123), ([0.00037653, 0.00037826], 122), ([0.00037826, 0.00037999], 121), ([0.00037999, 0.00038172], 120), ([0.00038172, 0.00038345], 119), ([0.00038345, 0.00038424], 118), ([0.00038424, 0.00038518], 117), ([0.00038518, 0.00038597], 116), ([0.00038597, 0.0003877], 115), ([0.0003877, 0.00038943], 114), ([0.00038943, 0.00039116], 113), ([0.00039116, 0.00039289], 112), ([0.00039289, 0.00039462], 111), ([0.00039462, 0.00039635], 110), ([0.00039635, 0.00039808], 109), ([0.00039808, 0.00039981], 108), ([0.00039981, 0.00040059], 107), ([0.00040059, 0.00040154], 106), ([0.00040154, 0.00040232], 105), ([0.00040232, 0.00040327], 104), ([0.00040327, 0.00040405], 103), ([0.00040405, 0.000405], 102), ([0.000405, 0.00040578], 101), ([0.00040578, 0.00040751], 100), ([0.00040751, 0.00040924], 99), ([0.00040924, 0.00041097], 98), ([0.00041097, 0.0004127], 97), ([0.0004127, 0.00041443], 96), ([0.00041443, 0.00041616], 95), ([0.00041616, 0.00041789], 94), ([0.00041789, 0.00041962], 93), ([0.00041962, 0.00042135], 92), ([0.00042135, 0.00042308], 91), ([0.00042308, 0.00042387], 90), ([0.00042387, 0.00042481], 89), ([0.00042481, 0.0004256], 88), ([0.0004256, 0.00042654], 87), ([0.00042654, 0.00042733], 86), ([0.00042733, 0.00042827], 85), ([0.00042827, 0.00042906], 84), ([0.00042906, 0.00043], 83), ([0.00043, 0.00043079], 82), ([0.00043079, 0.00043173], 81), ([0.00043173, 0.00043252], 80), ([0.00043252, 0.00043346], 79), ([0.00043346, 0.00043425], 78), ([0.00043425, 0.00043519], 77), ([0.00043519, 0.00043598], 76), ([0.00043598, 0.00043692], 75), ([0.00043692, 0.00043771], 74), ([0.00043771, 0.00043865], 73), ([0.00043865, 0.00043944], 72), ([0.00043944, 0.00044038], 71), ([0.00044038, 0.00044117], 70), ([0.00044117, 0.00044211], 69), ([0.00044211, 0.0004429], 68), ([0.0004429, 0.00044463], 67), ([0.00044463, 0.00044636], 66), ([0.00044636, 0.00044809], 65), ([0.00044809, 0.00044982], 64), ([0.00044982, 0.00045155], 63), ([0.00045155, 0.00045328], 62), ([0.00045328, 0.00045406], 61), ([0.00045406, 0.000455], 60), ([0.000455, 0.00045579], 59), ([0.00045579, 0.00045673], 58), ([0.00045673, 0.00045752], 57), ([0.00045752, 0.00045846], 56), ([0.00045846, 0.00045925], 55), ([0.00045925, 0.00046098], 54), ([0.00046098, 0.00046271], 53), ([0.00046271, 0.00046444], 52), ([0.00046444, 0.00046617], 51), ([0.00046617, 0.0004679], 50), ([0.0004679, 0.00046963], 49), ([0.00046963, 0.00047136], 48), ([0.00047136, 0.00047309], 47), ([0.00047309, 0.00047482], 46), ([0.00047482, 0.00047655], 45), ([0.00047655, 0.00047828], 44), ([0.00047828, 0.00048001], 43), ([0.00048001, 0.00048174], 42), ([0.00048174, 0.00048598], 41), ([0.00048598, 0.00048771], 40), ([0.00048771, 0.00048944], 39), ([0.00048944, 0.00049117], 38), ([0.00049117, 0.0004929], 37), ([0.0004929, 0.00049463], 36), ([0.00049463, 0.00049636], 35), ([0.00049636, 0.00049809], 34), ([0.00049809, 0.00049982], 33), ([0.00049982, 0.00050155], 32), ([0.00050155, 0.00050328], 31), ([0.00050328, 0.00050501], 30), ([0.00050501, 0.00050674], 29), ([0.00050674, 0.00050847], 28), ([0.00050847, 0.0005102], 27), ([0.0005102, 0.00051193], 26), ([0.00051193, 0.00053704], 25), ([0.00053704, 0.00053877], 24), ([0.00053877, 0.0005405], 23), ([0.0005405, 0.00054223], 22), ([0.00054223, 0.00054396], 21), ([0.00054396, 0.00057761], 20), ([0.00057761, 0.00057934], 19), ([0.00057934, 0.00058107], 18), ([0.00058107, 0.0005828], 17), ([0.0005828, 0.00058453], 16), ([0.00058453, 0.00058626], 15), ([0.00058626, 0.00058799], 14), ([0.00058799, 0.00058972], 13), ([0.00058972, 0.00059145], 12), ([0.00059145, 0.00059318], 11), ([0.00059318, 0.00059491], 10), ([0.00059491, 0.00062856], 9), ([0.00062856, 0.00063029], 8), ([0.00063029, 0.00063202], 7), ([0.00063202, 0.00063375], 6), ([0.00063375, 0.00063548], 5), ([0.00063548, 0.00068643], 4), ([0.00068643, 0.00074084], 3), ([0.00074084, 0.00074257], 2), ([0.00074257, 0.0007443], 1)]
        # self.draw_mode(from_octave_residuals_1, title='Мода для остатков канал 1')
        # from_octave_residuals_2 = [([-0.00053872, -0.00053531], 1), ([-0.00053531, -0.00050492], 2), ([-0.00050492, -0.00050151], 3), ([-0.00050151, -0.0004981], 4), ([-0.0004981, -0.00049644], 5), ([-0.00049644, -0.0004947], 6), ([-0.0004947, -0.00049462], 7), ([-0.00049462, -0.00049303], 8), ([-0.00049303, -0.00049129], 9), ([-0.00049129, -0.00049121], 10), ([-0.00049121, -0.00048962], 11), ([-0.00048962, -0.00048788], 12), ([-0.00048788, -0.0004878], 13), ([-0.0004878, -0.00048621], 14), ([-0.00048621, -0.00048447], 15), ([-0.00048447, -0.0004844], 16), ([-0.0004844, -0.00048281], 17), ([-0.00048281, -0.00048252], 18), ([-0.00048252, -0.00048106], 19), ([-0.00048106, -0.00048099], 20), ([-0.00048099, -0.0004794], 21), ([-0.0004794, -0.00047766], 22), ([-0.00047766, -0.00047758], 23), ([-0.00047758, -0.00047599], 24), ([-0.00047599, -0.00047425], 25), ([-0.00047425, -0.00047417], 26), ([-0.00047417, -0.00047258], 27), ([-0.00047258, -0.00047251], 28), ([-0.00047251, -0.00047084], 29), ([-0.00047084, -0.00047076], 30), ([-0.00047076, -0.00046917], 31), ([-0.00046917, -0.0004691], 32), ([-0.0004691, -0.00046751], 33), ([-0.00046751, -0.00046743], 34), ([-0.00046743, -0.00046736], 35), ([-0.00046736, -0.00046577], 36), ([-0.00046577, -0.00046569], 37), ([-0.00046569, -0.0004641], 38), ([-0.0004641, -0.00046402], 39), ([-0.00046402, -0.00046395], 40), ([-0.00046395, -0.00046236], 41), ([-0.00046236, -0.00046228], 42), ([-0.00046228, -0.00046069], 43), ([-0.00046069, -0.00046062], 44), ([-0.00046062, -0.00046054], 45), ([-0.00046054, -0.00045895], 46), ([-0.00045895, -0.00045887], 47), ([-0.00045887, -0.00045729], 48), ([-0.00045729, -0.00045721], 49), ([-0.00045721, -0.00045713], 50), ([-0.00045713, -0.00045554], 51), ([-0.00045554, -0.00045547], 52), ([-0.00045547, -0.00045388], 53), ([-0.00045388, -0.0004538], 54), ([-0.0004538, -0.00045372], 55), ([-0.00045372, -0.00045214], 56), ([-0.00045214, -0.00045206], 57), ([-0.00045206, -0.00045198], 58), ([-0.00045198, -0.00045047], 59), ([-0.00045047, -0.00045039], 60), ([-0.00045039, -0.00045032], 61), ([-0.00045032, -0.00044873], 62), ([-0.00044873, -0.00044865], 63), ([-0.00044865, -0.00044857], 64), ([-0.00044857, -0.00044706], 65), ([-0.00044706, -0.00044699], 66), ([-0.00044699, -0.00044691], 67), ([-0.00044691, -0.00044524], 68), ([-0.00044524, -0.00044517], 69), ([-0.00044517, -0.00044365], 70), ([-0.00044365, -0.00044358], 71), ([-0.00044358, -0.0004435], 72), ([-0.0004435, -0.00044183], 73), ([-0.00044183, -0.00044176], 74), ([-0.00044176, -0.00044025], 75), ([-0.00044025, -0.00044017], 76), ([-0.00044017, -0.00044009], 77), ([-0.00044009, -0.00043843], 78), ([-0.00043843, -0.00043835], 79), ([-0.00043835, -0.00043676], 80), ([-0.00043676, -0.00043668], 81), ([-0.00043668, -0.00043494], 82), ([-0.00043494, -0.00043335], 83), ([-0.00043335, -0.00043328], 84), ([-0.00043328, -0.00043153], 85), ([-0.00043153, -0.00042995], 86), ([-0.00042995, -0.00042987], 87), ([-0.00042987, -0.00042813], 88), ([-0.00042813, -0.00042654], 89), ([-0.00042654, -0.00042646], 90), ([-0.00042646, -0.00042633], 91), ([-0.00042633, -0.00042495], 92), ([-0.00042495, -0.00042472], 93), ([-0.00042472, -0.00042313], 94), ([-0.00042313, -0.00042305], 95), ([-0.00042305, -0.00042154], 96), ([-0.00042154, -0.00042131], 97), ([-0.00042131, -0.00041972], 98), ([-0.00041972, -0.00041965], 99), ([-0.00041965, -0.00041813], 100), ([-0.00041813, -0.0004179], 101), ([-0.0004179, -0.00041631], 102), ([-0.00041631, -0.00041624], 103), ([-0.00041624, -0.00041472], 104), ([-0.00041472, -0.0004145], 105), ([-0.0004145, -0.00041283], 106), ([-0.00041283, -0.00041132], 107), ([-0.00041132, -0.00041109], 108), ([-0.00041109, -0.00040942], 109), ([-0.00040942, -0.00040768], 110), ([-0.00040768, -0.00040601], 111), ([-0.00040601, -0.00040427], 112), ([-0.00040427, -0.00040261], 113), ([-0.00040261, -0.00040086], 114), ([-0.00040086, -0.0003992], 115), ([-0.0003992, -0.00039746], 116), ([-0.00039746, -0.00039579], 117), ([-0.00039579, -0.00039405], 118), ([-0.00039405, -0.00039399], 119), ([-0.00039399, -0.00039064], 120), ([-0.00039064, -0.00039059], 121), ([-0.00039059, -0.0003889], 122), ([-0.0003889, -0.00038723], 123), ([-0.00038723, -0.00038718], 124), ([-0.00038718, -0.00038549], 125), ([-0.00038549, -0.00038382], 126), ([-0.00038382, -0.00038377], 127), ([-0.00038377, -0.00038208], 128), ([-0.00038208, -0.00038042], 129), ([-0.00038042, -0.00038036], 130), ([-0.00038036, -0.00037898], 131), ([-0.00037898, -0.00037867], 132), ([-0.00037867, -0.00037701], 133), ([-0.00037701, -0.00037696], 134), ([-0.00037696, -0.00037557], 135), ([-0.00037557, -0.00037527], 136), ([-0.00037527, -0.0003736], 137), ([-0.0003736, -0.00037355], 138), ([-0.00037355, -0.00037216], 139), ([-0.00037216, -0.00037186], 140), ([-0.00037186, -0.00037019], 141), ([-0.00037019, -0.00037014], 142), ([-0.00037014, -0.00036876], 143), ([-0.00036876, -0.00036845], 144), ([-0.00036845, -0.00036678], 145), ([-0.00036678, -0.00036504], 146), ([-0.00036504, -0.00036338], 147), ([-0.00036338, -0.00036163], 148), ([-0.00036163, -0.00035997], 149), ([-0.00035997, -0.00035823], 150), ([-0.00035823, -0.00035656], 151), ([-0.00035656, -0.00035482], 152), ([-0.00035482, -0.00035315], 153), ([-0.00035315, -0.00035308], 154), ([-0.00035308, -0.00035141], 155), ([-0.00035141, -0.00034975], 156), ([-0.00034975, -0.00034967], 157), ([-0.00034967, -0.000348], 158), ([-0.000348, -0.00034634], 159), ([-0.00034634, -0.00034626], 160), ([-0.00034626, -0.0003446], 161), ([-0.0003446, -0.00034293], 162), ([-0.00034293, -0.00034285], 163), ([-0.00034285, -0.00034119], 164), ([-0.00034119, -0.00033952], 165), ([-0.00033952, -0.00033944], 166), ([-0.00033944, -0.00033778], 167), ([-0.00033778, -0.0003377], 168), ([-0.0003377, -0.00033611], 169), ([-0.00033611, -0.00033604], 170), ([-0.00033604, -0.00033437], 171), ([-0.00033437, -0.00033271], 172), ([-0.00033271, -0.00033263], 173), ([-0.00033263, -0.00033096], 174), ([-0.00033096, -0.00032922], 175), ([-0.00032922, -0.00032756], 176), ([-0.00032756, -0.00032581], 177), ([-0.00032581, -0.00032415], 178), ([-0.00032415, -0.00032241], 179), ([-0.00032241, -0.00032074], 180), ([-0.00032074, -0.000319], 181), ([-0.000319, -0.00031733], 182), ([-0.00031733, -0.00031559], 183), ([-0.00031559, -0.00031392], 184), ([-0.00031392, -0.00031218], 185), ([-0.00031218, -0.00031052], 186), ([-0.00031052, -0.00030877], 187), ([-0.00030877, -0.00030711], 188), ([-0.00030711, -0.00030537], 189), ([-0.00030537, -0.0003037], 190), ([-0.0003037, -0.00030196], 191), ([-0.00030196, -0.00030029], 192), ([-0.00030029, -0.00029855], 193), ([-0.00029855, -0.00029688], 194), ([-0.00029688, -0.00029514], 195), ([-0.00029514, -0.00029173], 196), ([-0.00029173, -0.00028833], 197), ([-0.00028833, -0.00028492], 198), ([-0.00028492, -0.00028151], 199), ([-0.00028151, 6.6283e-06], 200), ([6.6283e-06, 7.3641e-06], 199), ([7.3641e-06, 6.5492e-05], 198), ([6.5492e-05, 0.00012436], 197), ([0.00012436, 0.00017807], 196), ([0.00017807, 0.00017881], 195), ([0.00017881, 0.00017954], 194), ([0.00017954, 0.00018028], 193), ([0.00018028, 0.00018101], 192), ([0.00018101, 0.00018175], 191), ([0.00018175, 0.00018248], 190), ([0.00018248, 0.00018322], 189), ([0.00018322, 0.00023703], 188), ([0.00023703, 0.00028045], 187), ([0.00028045, 0.00028118], 186), ([0.00028118, 0.00028192], 185), ([0.00028192, 0.00028265], 184), ([0.00028265, 0.00028339], 183), ([0.00028339, 0.00028412], 182), ([0.00028412, 0.00028486], 181), ([0.00028486, 0.0002856], 180), ([0.0002856, 0.00028633], 179), ([0.00028633, 0.00028707], 178), ([0.00028707, 0.0002878], 177), ([0.0002878, 0.00028854], 176), ([0.00028854, 0.00028928], 175), ([0.00028928, 0.00029001], 174), ([0.00029001, 0.00029075], 173), ([0.00029075, 0.00029148], 172), ([0.00029148, 0.00029222], 171), ([0.00029222, 0.00029295], 170), ([0.00029295, 0.00029369], 169), ([0.00029369, 0.00029443], 168), ([0.00029443, 0.00029516], 167), ([0.00029516, 0.0002959], 166), ([0.0002959, 0.00031944], 165), ([0.00031944, 0.00032018], 164), ([0.00032018, 0.00032091], 163), ([0.00032091, 0.00032165], 162), ([0.00032165, 0.00032238], 161), ([0.00032238, 0.00032312], 160), ([0.00032312, 0.00032386], 159), ([0.00032386, 0.00032459], 158), ([0.00032459, 0.00032533], 157), ([0.00032533, 0.00032606], 156), ([0.00032606, 0.0003268], 155), ([0.0003268, 0.00032754], 154), ([0.00032754, 0.00032827], 153), ([0.00032827, 0.00032901], 152), ([0.00032901, 0.00032974], 151), ([0.00032974, 0.00033048], 150), ([0.00033048, 0.00033122], 149), ([0.00033122, 0.00033195], 148), ([0.00033195, 0.00033269], 147), ([0.00033269, 0.00033342], 146), ([0.00033342, 0.00033416], 145), ([0.00033416, 0.00033489], 144), ([0.00033489, 0.00033563], 143), ([0.00033563, 0.00033637], 142), ([0.00033637, 0.0003371], 141), ([0.0003371, 0.00033784], 140), ([0.00033784, 0.00033857], 139), ([0.00033857, 0.00033931], 138), ([0.00033931, 0.00035255], 137), ([0.00035255, 0.00035329], 136), ([0.00035329, 0.00035402], 135), ([0.00035402, 0.00035476], 134), ([0.00035476, 0.00035549], 133), ([0.00035549, 0.00035623], 132), ([0.00035623, 0.00035697], 131), ([0.00035697, 0.0003577], 130), ([0.0003577, 0.00035844], 129), ([0.00035844, 0.00035917], 128), ([0.00035917, 0.00035991], 127), ([0.00035991, 0.00036065], 126), ([0.00036065, 0.00036138], 125), ([0.00036138, 0.00036212], 124), ([0.00036212, 0.00036285], 123), ([0.00036285, 0.00036359], 122), ([0.00036359, 0.00036432], 121), ([0.00036432, 0.00036506], 120), ([0.00036506, 0.0003658], 119), ([0.0003658, 0.00036653], 118), ([0.00036653, 0.00036727], 117), ([0.00036727, 0.000368], 116), ([0.000368, 0.00036874], 115), ([0.00036874, 0.00036948], 114), ([0.00036948, 0.00037021], 113), ([0.00037021, 0.00037095], 112), ([0.00037095, 0.00037168], 111), ([0.00037168, 0.00037242], 110), ([0.00037242, 0.00037315], 109), ([0.00037315, 0.00037389], 108), ([0.00037389, 0.00037463], 107), ([0.00037463, 0.00037536], 106), ([0.00037536, 0.0003761], 105), ([0.0003761, 0.00037683], 104), ([0.00037683, 0.00037757], 103), ([0.00037757, 0.00037831], 102), ([0.00037831, 0.00039008], 101), ([0.00039008, 0.00039081], 100), ([0.00039081, 0.00039155], 99), ([0.00039155, 0.00039228], 98), ([0.00039228, 0.00039302], 97), ([0.00039302, 0.00039375], 96), ([0.00039375, 0.00039449], 95), ([0.00039449, 0.00039523], 94), ([0.00039523, 0.00039596], 93), ([0.00039596, 0.0003967], 92), ([0.0003967, 0.00039743], 91), ([0.00039743, 0.00039817], 90), ([0.00039817, 0.00039891], 89), ([0.00039891, 0.00039964], 88), ([0.00039964, 0.00040038], 87), ([0.00040038, 0.00040111], 86), ([0.00040111, 0.00040185], 85), ([0.00040185, 0.00040258], 84), ([0.00040258, 0.00040332], 83), ([0.00040332, 0.00040406], 82), ([0.00040406, 0.00040479], 81), ([0.00040479, 0.00040553], 80), ([0.00040553, 0.00040626], 79), ([0.00040626, 0.000407], 78), ([0.000407, 0.00040774], 77), ([0.00040774, 0.00040847], 76), ([0.00040847, 0.00040921], 75), ([0.00040921, 0.00040994], 74), ([0.00040994, 0.00041068], 73), ([0.00041068, 0.00041141], 72), ([0.00041141, 0.00044158], 71), ([0.00044158, 0.00044232], 70), ([0.00044232, 0.00044305], 69), ([0.00044305, 0.00044379], 68), ([0.00044379, 0.00044452], 67), ([0.00044452, 0.00044526], 66), ([0.00044526, 0.000446], 65), ([0.000446, 0.00044673], 64), ([0.00044673, 0.00044747], 63), ([0.00044747, 0.0004482], 62), ([0.0004482, 0.00044894], 61), ([0.00044894, 0.00048131], 60), ([0.00048131, 0.00048205], 59), ([0.00048205, 0.00048278], 58), ([0.00048278, 0.00048352], 57), ([0.00048352, 0.00048426], 56), ([0.00048426, 0.00048499], 55), ([0.00048499, 0.00048573], 54), ([0.00048573, 0.00048646], 53), ([0.00048646, 0.0004872], 52), ([0.0004872, 0.00048794], 51), ([0.00048794, 0.00048867], 50), ([0.00048867, 0.00048941], 49), ([0.00048941, 0.00049014], 48), ([0.00049014, 0.00049088], 47), ([0.00049088, 0.00049161], 46), ([0.00049161, 0.00049235], 45), ([0.00049235, 0.00049309], 44), ([0.00049309, 0.00049382], 43), ([0.00049382, 0.00049456], 42), ([0.00049456, 0.00049529], 41), ([0.00049529, 0.00049603], 40), ([0.00049603, 0.00049677], 39), ([0.00049677, 0.0004975], 38), ([0.0004975, 0.00049824], 37), ([0.00049824, 0.00049897], 36), ([0.00049897, 0.00049971], 35), ([0.00049971, 0.00050045], 34), ([0.00050045, 0.00052988], 33), ([0.00052988, 0.00053061], 32), ([0.00053061, 0.00053135], 31), ([0.00053135, 0.00053208], 30), ([0.00053208, 0.00053282], 29), ([0.00053282, 0.00053355], 28), ([0.00053355, 0.00053429], 27), ([0.00053429, 0.00053503], 26), ([0.00053503, 0.00053576], 25), ([0.00053576, 0.0005365], 24), ([0.0005365, 0.00053723], 23), ([0.00053723, 0.00053797], 22), ([0.00053797, 0.00053871], 21), ([0.00053871, 0.00053944], 20), ([0.00053944, 0.00054018], 19), ([0.00054018, 0.00058285], 18), ([0.00058285, 0.00058359], 17), ([0.00058359, 0.00058432], 16), ([0.00058432, 0.00058506], 15), ([0.00058506, 0.0005858], 14), ([0.0005858, 0.00058653], 13), ([0.00058653, 0.00058727], 12), ([0.00058727, 0.000588], 11), ([0.000588, 0.00058874], 10), ([0.00058874, 0.00063877], 9), ([0.00063877, 0.00063951], 8), ([0.00063951, 0.00064025], 7), ([0.00064025, 0.00064098], 6), ([0.00064098, 0.00064172], 5), ([0.00064172, 0.00069543], 4), ([0.00069543, 0.00069617], 3), ([0.00069617, 0.0006969], 2), ([0.0006969, 0.00069764], 1)]
        # self.draw_mode(from_octave_residuals_2, title='Мода для остатков канал 2')


        # intervals_residuals = self.get_residuals(self.data_intervals, self.edge_points,
        #                                          self.intervals_regression_params_3)
        # self.draw_data_status_with_points(intervals_residuals, 'Остатки после вычитания кусочных регрессий',
        #                                   'Диаграмма статусов после вычитания кусочных регрессий')


        intervals_regression_params_3_1 = self.intervals_regression_params_3.copy()
        intervals_regression_params_3_1[0][2] = intervals_regression_params_3_1[0][1]
        intervals_regression_params_3_1[1][2] = intervals_regression_params_3_1[1][1]
        intervals_residuals_1 = self.get_residuals_1(self.data_intervals, [[self.intervals_regression_params_3[0][1]],
                                                                           [self.intervals_regression_params_3[1][1]]])
        # self.draw_data_status_with_points(intervals_residuals_1, '', '')

        for num_, res_list in enumerate(intervals_residuals_1):
            point = self.edge_points[num_]
            residuals = intervals_residuals_1[num_][point[0]:point[1]]
            inter_w = self.get_intersections_wrong_int(residuals)
            print(inter_w.points)
            inters = self.get_intersections(residuals)
            new_res_list = self.regularization(res_list, point[0], inters, True,
                                               f'Веса регуляризации, канал {num_ + 1}', '')
            self.get_octave_intervals(new_res_list)
            # infls, intersection = self.get_influences(new_res_list, inters)
            # self.draw_residuals(new_res_list, intersection, title=f'Остатки после регуляризации, канал {num_ + 1}')
            # m_l = max([res[0] for res in infls])
            # self.draw_params([infl[0] for infl in infls], title=f'Размах, канал {num_ + 1}', param_name='l')
            # self.draw_params([infl[1] for infl in infls], title=f'Относительный остаток, канал {num_ + 1}',
            #                  param_name='|r|', params_2=[infl[0] for infl in infls], param_name_2='1-l')
            # fig_, ax_ = self.draw_data_status_template([0, max(m_l, 2)],
            #                                            title=f'Диаграмма статусов после регуляризации, канал{num_ + 1}')
            # for infl in infls:
            #     self.add_point(infl, ax_)
            # fig_.show()

        # from_octave_reg_1 = [([-0.00070094, -0.00059375], 1), ([-0.00059375, -0.00058774], 2), ([-0.00058774, -0.0005286], 3), ([-0.0005286, -0.0005226], 4), ([-0.0005226, -0.00051659], 5), ([-0.00051659, -0.00051059], 6), ([-0.00051059, -0.00050969], 7), ([-0.00050969, -0.00050541], 8), ([-0.00050541, -0.00050458], 9), ([-0.00050458, -0.00050113], 10), ([-0.00050113, -0.00050086], 11), ([-0.00050086, -0.00049857], 12), ([-0.00049857, -0.00049686], 13), ([-0.00049686, -0.00049659], 14), ([-0.00049659, -0.00049258], 15), ([-0.00049258, -0.00049257], 16), ([-0.00049257, -0.00049231], 17), ([-0.00049231, -0.0004883], 18), ([-0.0004883, -0.00048803], 19), ([-0.00048803, -0.00048776], 20), ([-0.00048776, -0.00048656], 21), ([-0.00048656, -0.00048403], 22), ([-0.00048403, -0.00048376], 23), ([-0.00048376, -0.00048349], 24), ([-0.00048349, -0.00048055], 25), ([-0.00048055, -0.00047975], 26), ([-0.00047975, -0.00047948], 27), ([-0.00047948, -0.00047921], 28), ([-0.00047921, -0.00047548], 29), ([-0.00047548, -0.0004752], 30), ([-0.0004752, -0.00047493], 31), ([-0.00047493, -0.00047137], 32), ([-0.00047137, -0.0004712], 33), ([-0.0004712, -0.00047093], 34), ([-0.00047093, -0.00047066], 35), ([-0.00047066, -0.00046709], 36), ([-0.00046709, -0.00046692], 37), ([-0.00046692, -0.00046665], 38), ([-0.00046665, -0.00046638], 39), ([-0.00046638, -0.00046282], 40), ([-0.00046282, -0.00046265], 41), ([-0.00046265, -0.00046237], 42), ([-0.00046237, -0.0004621], 43), ([-0.0004621, -0.00045881], 44), ([-0.00045881, -0.00045854], 45), ([-0.00045854, -0.00045837], 46), ([-0.00045837, -0.0004581], 47), ([-0.0004581, -0.00045783], 48), ([-0.00045783, -0.00045756], 49), ([-0.00045756, -0.00045453], 50), ([-0.00045453, -0.00045426], 51), ([-0.00045426, -0.00045409], 52), ([-0.00045409, -0.00045382], 53), ([-0.00045382, -0.00045355], 54), ([-0.00045355, -0.00045328], 55), ([-0.00045328, -0.00045026], 56), ([-0.00045026, -0.00044982], 57), ([-0.00044982, -0.00044955], 58), ([-0.00044955, -0.00044927], 59), ([-0.00044927, -0.000449], 60), ([-0.000449, -0.00044598], 61), ([-0.00044598, -0.00044554], 62), ([-0.00044554, -0.00044527], 63), ([-0.00044527, -0.000445], 64), ([-0.000445, -0.00044473], 65), ([-0.00044473, -0.00044171], 66), ([-0.00044171, -0.00044099], 67), ([-0.00044099, -0.00044072], 68), ([-0.00044072, -0.00044045], 69), ([-0.00044045, -0.00043743], 70), ([-0.00043743, -0.00043672], 71), ([-0.00043672, -0.00043644], 72), ([-0.00043644, -0.00043617], 73), ([-0.00043617, -0.00043315], 74), ([-0.00043315, -0.00043244], 75), ([-0.00043244, -0.00043217], 76), ([-0.00043217, -0.0004319], 77), ([-0.0004319, -0.00042888], 78), ([-0.00042888, -0.00042789], 79), ([-0.00042789, -0.00042762], 80), ([-0.00042762, -0.00042742], 81), ([-0.00042742, -0.0004246], 82), ([-0.0004246, -0.00042362], 83), ([-0.00042362, -0.00042334], 84), ([-0.00042334, -0.00042142], 85), ([-0.00042142, -0.00042059], 86), ([-0.00042059, -0.00042032], 87), ([-0.00042032, -0.00041934], 88), ([-0.00041934, -0.00041907], 89), ([-0.00041907, -0.00041632], 90), ([-0.00041632, -0.00041605], 91), ([-0.00041605, -0.00041541], 92), ([-0.00041541, -0.00041506], 93), ([-0.00041506, -0.00041479], 94), ([-0.00041479, -0.00041425], 95), ([-0.00041425, -0.00041204], 96), ([-0.00041204, -0.00041079], 97), ([-0.00041079, -0.00041051], 98), ([-0.00041051, -0.00041024], 99), ([-0.00041024, -0.00040997], 100), ([-0.00040997, -0.00040776], 101), ([-0.00040776, -0.00040651], 102), ([-0.00040651, -0.00040624], 103), ([-0.00040624, -0.00040597], 104), ([-0.00040597, -0.0004057], 105), ([-0.0004057, -0.00040349], 106), ([-0.00040349, -0.00040223], 107), ([-0.00040223, -0.00040196], 108), ([-0.00040196, -0.00040169], 109), ([-0.00040169, -0.00040142], 110), ([-0.00040142, -0.00039769], 111), ([-0.00039769, -0.00039741], 112), ([-0.00039741, -0.00039714], 113), ([-0.00039714, -0.00039341], 114), ([-0.00039341, -0.00039314], 115), ([-0.00039314, -0.00039287], 116), ([-0.00039287, -0.00038913], 117), ([-0.00038913, -0.00038886], 118), ([-0.00038886, -0.00038859], 119), ([-0.00038859, -0.00038486], 120), ([-0.00038486, -0.00038458], 121), ([-0.00038458, -0.00038431], 122), ([-0.00038431, -0.00038058], 123), ([-0.00038058, -0.00038031], 124), ([-0.00038031, -0.00038004], 125), ([-0.00038004, -0.0003763], 126), ([-0.0003763, -0.00037603], 127), ([-0.00037603, -0.00037576], 128), ([-0.00037576, -0.00037549], 129), ([-0.00037549, -0.00037522], 130), ([-0.00037522, -0.00037203], 131), ([-0.00037203, -0.00037176], 132), ([-0.00037176, -0.00037148], 133), ([-0.00037148, -0.00037121], 134), ([-0.00037121, -0.00037094], 135), ([-0.00037094, -0.00036775], 136), ([-0.00036775, -0.00036748], 137), ([-0.00036748, -0.00036721], 138), ([-0.00036721, -0.00036694], 139), ([-0.00036694, -0.00036667], 140), ([-0.00036667, -0.00036527], 141), ([-0.00036527, -0.00036347], 142), ([-0.00036347, -0.0003632], 143), ([-0.0003632, -0.00036293], 144), ([-0.00036293, -0.00036266], 145), ([-0.00036266, -0.00036239], 146), ([-0.00036239, -0.00036228], 147), ([-0.00036228, -0.0003592], 148), ([-0.0003592, -0.00035893], 149), ([-0.00035893, -0.00035865], 150), ([-0.00035865, -0.00035838], 151), ([-0.00035838, -0.00035811], 152), ([-0.00035811, -0.00035627], 153), ([-0.00035627, -0.00035492], 154), ([-0.00035492, -0.00035438], 155), ([-0.00035438, -0.00035411], 156), ([-0.00035411, -0.00035384], 157), ([-0.00035384, -0.00035027], 158), ([-0.00035027, -0.0003501], 159), ([-0.0003501, -0.00034983], 160), ([-0.00034983, -0.00034956], 161), ([-0.00034956, -0.00034583], 162), ([-0.00034583, -0.00034555], 163), ([-0.00034555, -0.00034528], 164), ([-0.00034528, -0.00034426], 165), ([-0.00034426, -0.00034155], 166), ([-0.00034155, -0.00034128], 167), ([-0.00034128, -0.00034101], 168), ([-0.00034101, -0.00033826], 169), ([-0.00033826, -0.00033727], 170), ([-0.00033727, -0.000337], 171), ([-0.000337, -0.00033673], 172), ([-0.00033673, -0.000333], 173), ([-0.000333, -0.00033272], 174), ([-0.00033272, -0.00033245], 175), ([-0.00033245, -0.00033225], 176), ([-0.00033225, -0.00032872], 177), ([-0.00032872, -0.00032845], 178), ([-0.00032845, -0.00032818], 179), ([-0.00032818, -0.00032624], 180), ([-0.00032624, -0.00032444], 181), ([-0.00032444, -0.00032417], 182), ([-0.00032417, -0.0003239], 183), ([-0.0003239, -0.00032024], 184), ([-0.00032024, -0.00032017], 185), ([-0.00032017, -0.0003199], 186), ([-0.0003199, -0.00031962], 187), ([-0.00031962, -0.0003185], 188), ([-0.0003185, -0.00031535], 189), ([-0.00031535, -0.00031423], 190), ([-0.00031423, -0.00031422], 191), ([-0.00031422, -0.00031107], 192), ([-0.00031107, -0.00030995], 193), ([-0.00030995, -0.00030679], 194), ([-0.00030679, -0.00030252], 195), ([-0.00030252, -0.00029824], 196), ([-0.00029824, -0.00029397], 197), ([-0.00029397, -0.00028969], 198), ([-0.00028969, 0.00029193], 200), ([0.00029193, 0.00029366], 175), ([0.00029366, 0.00029539], 174), ([0.00029539, 0.00029712], 173), ([0.00029712, 0.00029885], 172), ([0.00029885, 0.00030058], 171), ([0.00030058, 0.00030231], 170), ([0.00030231, 0.00030404], 169), ([0.00030404, 0.00030577], 168), ([0.00030577, 0.0003075], 167), ([0.0003075, 0.00030923], 166), ([0.00030923, 0.00031096], 165), ([0.00031096, 0.00031269], 164), ([0.00031269, 0.00031442], 163), ([0.00031442, 0.00031615], 162), ([0.00031615, 0.00031788], 161), ([0.00031788, 0.00031961], 160), ([0.00031961, 0.00032134], 159), ([0.00032134, 0.00032307], 158), ([0.00032307, 0.0003248], 157), ([0.0003248, 0.00032653], 156), ([0.00032653, 0.00032731], 155), ([0.00032731, 0.00032904], 154), ([0.00032904, 0.00033077], 153), ([0.00033077, 0.0003325], 152), ([0.0003325, 0.00033253], 151), ([0.00033253, 0.00033423], 150), ([0.00033423, 0.00033596], 149), ([0.00033596, 0.00033769], 148), ([0.00033769, 0.00033942], 147), ([0.00033942, 0.00034115], 146), ([0.00034115, 0.00034288], 145), ([0.00034288, 0.00034461], 144), ([0.00034461, 0.00034634], 143), ([0.00034634, 0.00034713], 142), ([0.00034713, 0.00034807], 141), ([0.00034807, 0.00034886], 140), ([0.00034886, 0.0003498], 139), ([0.0003498, 0.00035059], 138), ([0.00035059, 0.00035232], 137), ([0.00035232, 0.00035405], 136), ([0.00035405, 0.00035578], 135), ([0.00035578, 0.00035751], 134), ([0.00035751, 0.00035924], 133), ([0.00035924, 0.00036096], 132), ([0.00036096, 0.00036269], 131), ([0.00036269, 0.00036442], 130), ([0.00036442, 0.00036615], 129), ([0.00036615, 0.00036788], 128), ([0.00036788, 0.00036961], 127), ([0.00036961, 0.00037134], 126), ([0.00037134, 0.00037307], 125), ([0.00037307, 0.0003748], 124), ([0.0003748, 0.00037653], 123), ([0.00037653, 0.00037826], 122), ([0.00037826, 0.00037999], 121), ([0.00037999, 0.00038172], 120), ([0.00038172, 0.00038345], 119), ([0.00038345, 0.00038424], 118), ([0.00038424, 0.00038518], 117), ([0.00038518, 0.00038597], 116), ([0.00038597, 0.0003877], 115), ([0.0003877, 0.00038943], 114), ([0.00038943, 0.00039116], 113), ([0.00039116, 0.00039289], 112), ([0.00039289, 0.00039462], 111), ([0.00039462, 0.00039635], 110), ([0.00039635, 0.00039808], 109), ([0.00039808, 0.00039981], 108), ([0.00039981, 0.00040059], 107), ([0.00040059, 0.00040154], 106), ([0.00040154, 0.00040232], 105), ([0.00040232, 0.00040327], 104), ([0.00040327, 0.00040405], 103), ([0.00040405, 0.000405], 102), ([0.000405, 0.00040578], 101), ([0.00040578, 0.00040751], 100), ([0.00040751, 0.00040924], 99), ([0.00040924, 0.00041097], 98), ([0.00041097, 0.0004127], 97), ([0.0004127, 0.00041443], 96), ([0.00041443, 0.00041616], 95), ([0.00041616, 0.00041789], 94), ([0.00041789, 0.00041962], 93), ([0.00041962, 0.00042135], 92), ([0.00042135, 0.00042308], 91), ([0.00042308, 0.00042387], 90), ([0.00042387, 0.00042481], 89), ([0.00042481, 0.0004256], 88), ([0.0004256, 0.00042654], 87), ([0.00042654, 0.00042733], 86), ([0.00042733, 0.00042827], 85), ([0.00042827, 0.00042906], 84), ([0.00042906, 0.00043], 83), ([0.00043, 0.00043079], 82), ([0.00043079, 0.00043173], 81), ([0.00043173, 0.00043252], 80), ([0.00043252, 0.00043346], 79), ([0.00043346, 0.00043425], 78), ([0.00043425, 0.00043519], 77), ([0.00043519, 0.00043598], 76), ([0.00043598, 0.00043692], 75), ([0.00043692, 0.00043771], 74), ([0.00043771, 0.00043865], 73), ([0.00043865, 0.00043944], 72), ([0.00043944, 0.00044038], 71), ([0.00044038, 0.00044117], 70), ([0.00044117, 0.00044211], 69), ([0.00044211, 0.0004429], 68), ([0.0004429, 0.00044463], 67), ([0.00044463, 0.00044636], 66), ([0.00044636, 0.00044809], 65), ([0.00044809, 0.00044982], 64), ([0.00044982, 0.00045155], 63), ([0.00045155, 0.00045328], 62), ([0.00045328, 0.00045406], 61), ([0.00045406, 0.000455], 60), ([0.000455, 0.00045579], 59), ([0.00045579, 0.00045673], 58), ([0.00045673, 0.00045752], 57), ([0.00045752, 0.00045846], 56), ([0.00045846, 0.00045925], 55), ([0.00045925, 0.00046098], 54), ([0.00046098, 0.00046271], 53), ([0.00046271, 0.00046444], 52), ([0.00046444, 0.00046617], 51), ([0.00046617, 0.0004679], 50), ([0.0004679, 0.00046963], 49), ([0.00046963, 0.00047136], 48), ([0.00047136, 0.00047309], 47), ([0.00047309, 0.00047482], 46), ([0.00047482, 0.00047655], 45), ([0.00047655, 0.00047828], 44), ([0.00047828, 0.00048001], 43), ([0.00048001, 0.00048174], 42), ([0.00048174, 0.00048598], 41), ([0.00048598, 0.00048771], 40), ([0.00048771, 0.00048944], 39), ([0.00048944, 0.00049117], 38), ([0.00049117, 0.0004929], 37), ([0.0004929, 0.00049463], 36), ([0.00049463, 0.00049636], 35), ([0.00049636, 0.00049809], 34), ([0.00049809, 0.00049982], 33), ([0.00049982, 0.00050155], 32), ([0.00050155, 0.00050328], 31), ([0.00050328, 0.00050501], 30), ([0.00050501, 0.00050674], 29), ([0.00050674, 0.00050847], 28), ([0.00050847, 0.0005102], 27), ([0.0005102, 0.00051193], 26), ([0.00051193, 0.00053704], 25), ([0.00053704, 0.00053877], 24), ([0.00053877, 0.0005405], 23), ([0.0005405, 0.00054223], 22), ([0.00054223, 0.00054396], 21), ([0.00054396, 0.00057761], 20), ([0.00057761, 0.00057934], 19), ([0.00057934, 0.00058107], 18), ([0.00058107, 0.0005828], 17), ([0.0005828, 0.00058453], 16), ([0.00058453, 0.00058626], 15), ([0.00058626, 0.00058799], 14), ([0.00058799, 0.00058972], 13), ([0.00058972, 0.00059145], 12), ([0.00059145, 0.00059318], 11), ([0.00059318, 0.00059491], 10), ([0.00059491, 0.00062856], 9), ([0.00062856, 0.00063029], 8), ([0.00063029, 0.00063202], 7), ([0.00063202, 0.00063375], 6), ([0.00063375, 0.00063548], 5), ([0.00063548, 0.00068643], 4), ([0.00068643, 0.00074084], 3), ([0.00074084, 0.00074257], 2), ([0.00074257, 0.0007443], 1)]
        # self.draw_mode(from_octave_reg_1, title='Мода после регуляризации канал 1')
        # from_octave_reg_2 = [([-0.00081253, -0.00080839], 1), ([-0.00080839, -0.00069748], 2), ([-0.00069748, -0.00058242], 3), ([-0.00058242, -0.00050492], 4), ([-0.00050492, -0.00050151], 5), ([-0.00050151, -0.0004981], 6), ([-0.0004981, -0.00049644], 7), ([-0.00049644, -0.00049637], 8), ([-0.00049637, -0.0004947], 9), ([-0.0004947, -0.00049462], 10), ([-0.00049462, -0.00049303], 11), ([-0.00049303, -0.00049223], 12), ([-0.00049223, -0.00049129], 13), ([-0.00049129, -0.00049121], 14), ([-0.00049121, -0.00048962], 15), ([-0.00048962, -0.00048808], 16), ([-0.00048808, -0.00048788], 17), ([-0.00048788, -0.0004878], 18), ([-0.0004878, -0.00048621], 19), ([-0.00048621, -0.00048447], 20), ([-0.00048447, -0.0004844], 21), ([-0.0004844, -0.00048394], 22), ([-0.00048394, -0.00048281], 23), ([-0.00048281, -0.00048106], 24), ([-0.00048106, -0.00048099], 25), ([-0.00048099, -0.0004798], 26), ([-0.0004798, -0.0004794], 27), ([-0.0004794, -0.00047766], 28), ([-0.00047766, -0.00047758], 29), ([-0.00047758, -0.00047599], 30), ([-0.00047599, -0.00047565], 31), ([-0.00047565, -0.00047425], 32), ([-0.00047425, -0.00047417], 33), ([-0.00047417, -0.00047258], 34), ([-0.00047258, -0.00047251], 35), ([-0.00047251, -0.00047151], 36), ([-0.00047151, -0.00047084], 37), ([-0.00047084, -0.00047076], 38), ([-0.00047076, -0.00046917], 39), ([-0.00046917, -0.0004691], 40), ([-0.0004691, -0.00046751], 41), ([-0.00046751, -0.00046743], 42), ([-0.00046743, -0.00046736], 43), ([-0.00046736, -0.00046736], 44), ([-0.00046736, -0.00046577], 45), ([-0.00046577, -0.00046569], 46), ([-0.00046569, -0.0004641], 47), ([-0.0004641, -0.00046402], 48), ([-0.00046402, -0.00046395], 49), ([-0.00046395, -0.00046236], 50), ([-0.00046236, -0.00046228], 51), ([-0.00046228, -0.00046069], 52), ([-0.00046069, -0.00046062], 53), ([-0.00046062, -0.00046054], 54), ([-0.00046054, -0.00045895], 55), ([-0.00045895, -0.00045887], 56), ([-0.00045887, -0.00045729], 57), ([-0.00045729, -0.00045721], 58), ([-0.00045721, -0.00045713], 59), ([-0.00045713, -0.00045554], 60), ([-0.00045554, -0.00045547], 61), ([-0.00045547, -0.00045388], 62), ([-0.00045388, -0.0004538], 63), ([-0.0004538, -0.00045372], 64), ([-0.00045372, -0.00045214], 65), ([-0.00045214, -0.00045206], 66), ([-0.00045206, -0.00045198], 67), ([-0.00045198, -0.00045047], 68), ([-0.00045047, -0.00045039], 69), ([-0.00045039, -0.00045032], 70), ([-0.00045032, -0.00044873], 71), ([-0.00044873, -0.00044865], 72), ([-0.00044865, -0.00044857], 73), ([-0.00044857, -0.00044706], 74), ([-0.00044706, -0.00044699], 75), ([-0.00044699, -0.00044691], 76), ([-0.00044691, -0.00044524], 77), ([-0.00044524, -0.00044517], 78), ([-0.00044517, -0.00044365], 79), ([-0.00044365, -0.00044358], 80), ([-0.00044358, -0.0004435], 81), ([-0.0004435, -0.00044183], 82), ([-0.00044183, -0.00044176], 83), ([-0.00044176, -0.00044025], 84), ([-0.00044025, -0.00044017], 85), ([-0.00044017, -0.00044009], 86), ([-0.00044009, -0.00043843], 87), ([-0.00043843, -0.00043835], 88), ([-0.00043835, -0.00043676], 89), ([-0.00043676, -0.00043668], 90), ([-0.00043668, -0.00043494], 91), ([-0.00043494, -0.00043335], 92), ([-0.00043335, -0.00043328], 93), ([-0.00043328, -0.00043153], 94), ([-0.00043153, -0.00042995], 95), ([-0.00042995, -0.00042987], 96), ([-0.00042987, -0.00042813], 97), ([-0.00042813, -0.00042654], 98), ([-0.00042654, -0.00042646], 99), ([-0.00042646, -0.00042495], 100), ([-0.00042495, -0.00042472], 101), ([-0.00042472, -0.00042313], 102), ([-0.00042313, -0.00042305], 103), ([-0.00042305, -0.00042154], 104), ([-0.00042154, -0.00042131], 105), ([-0.00042131, -0.00041972], 106), ([-0.00041972, -0.00041965], 107), ([-0.00041965, -0.00041813], 108), ([-0.00041813, -0.0004179], 109), ([-0.0004179, -0.00041631], 110), ([-0.00041631, -0.00041624], 111), ([-0.00041624, -0.00041472], 112), ([-0.00041472, -0.0004145], 113), ([-0.0004145, -0.00041283], 114), ([-0.00041283, -0.00041132], 115), ([-0.00041132, -0.00041109], 116), ([-0.00041109, -0.00040942], 117), ([-0.00040942, -0.00040768], 118), ([-0.00040768, -0.00040601], 119), ([-0.00040601, -0.00040427], 120), ([-0.00040427, -0.00040261], 121), ([-0.00040261, -0.00040086], 122), ([-0.00040086, -0.0003992], 123), ([-0.0003992, -0.00039746], 124), ([-0.00039746, -0.00039579], 125), ([-0.00039579, -0.00039405], 126), ([-0.00039405, -0.00039064], 127), ([-0.00039064, -0.0003889], 128), ([-0.0003889, -0.00038723], 129), ([-0.00038723, -0.00038549], 130), ([-0.00038549, -0.00038382], 131), ([-0.00038382, -0.00038208], 132), ([-0.00038208, -0.00038111], 133), ([-0.00038111, -0.00038042], 134), ([-0.00038042, -0.00037898], 135), ([-0.00037898, -0.00037867], 136), ([-0.00037867, -0.00037701], 137), ([-0.00037701, -0.00037557], 138), ([-0.00037557, -0.00037527], 139), ([-0.00037527, -0.0003736], 140), ([-0.0003736, -0.00037216], 141), ([-0.00037216, -0.00037186], 142), ([-0.00037186, -0.00037019], 143), ([-0.00037019, -0.00036876], 144), ([-0.00036876, -0.00036845], 145), ([-0.00036845, -0.00036678], 146), ([-0.00036678, -0.00036504], 147), ([-0.00036504, -0.00036338], 148), ([-0.00036338, -0.00036163], 149), ([-0.00036163, -0.00035997], 150), ([-0.00035997, -0.00035823], 151), ([-0.00035823, -0.00035656], 152), ([-0.00035656, -0.00035482], 153), ([-0.00035482, -0.00035315], 154), ([-0.00035315, -0.00035308], 155), ([-0.00035308, -0.00035141], 156), ([-0.00035141, -0.00034975], 157), ([-0.00034975, -0.00034967], 158), ([-0.00034967, -0.000348], 159), ([-0.000348, -0.00034634], 160), ([-0.00034634, -0.00034626], 161), ([-0.00034626, -0.0003446], 162), ([-0.0003446, -0.00034293], 163), ([-0.00034293, -0.00034285], 164), ([-0.00034285, -0.00034119], 165), ([-0.00034119, -0.00033952], 166), ([-0.00033952, -0.00033944], 167), ([-0.00033944, -0.00033778], 168), ([-0.00033778, -0.00033611], 169), ([-0.00033611, -0.00033604], 170), ([-0.00033604, -0.00033437], 171), ([-0.00033437, -0.00033271], 172), ([-0.00033271, -0.00033263], 173), ([-0.00033263, -0.00033096], 174), ([-0.00033096, -0.00032922], 175), ([-0.00032922, -0.00032756], 176), ([-0.00032756, -0.00032581], 177), ([-0.00032581, -0.00032415], 178), ([-0.00032415, -0.00032241], 179), ([-0.00032241, -0.00032074], 180), ([-0.00032074, -0.000319], 181), ([-0.000319, -0.00031733], 182), ([-0.00031733, -0.00031559], 183), ([-0.00031559, -0.00031392], 184), ([-0.00031392, -0.00031218], 185), ([-0.00031218, -0.00031052], 186), ([-0.00031052, -0.00030877], 187), ([-0.00030877, -0.00030711], 188), ([-0.00030711, -0.00030537], 189), ([-0.00030537, -0.0003037], 190), ([-0.0003037, -0.00030196], 191), ([-0.00030196, -0.00030029], 192), ([-0.00030029, -0.00029855], 193), ([-0.00029855, -0.00029688], 194), ([-0.00029688, -0.00029514], 195), ([-0.00029514, -0.00029173], 196), ([-0.00029173, -0.00028833], 197), ([-0.00028833, -0.00028492], 198), ([-0.00028492, 0.00028045], 200), ([0.00028045, 0.00028118], 186), ([0.00028118, 0.00028192], 185), ([0.00028192, 0.00028265], 184), ([0.00028265, 0.00028339], 183), ([0.00028339, 0.00028412], 182), ([0.00028412, 0.00028486], 181), ([0.00028486, 0.0002856], 180), ([0.0002856, 0.00028633], 179), ([0.00028633, 0.00028707], 178), ([0.00028707, 0.0002878], 177), ([0.0002878, 0.00028854], 176), ([0.00028854, 0.00028928], 175), ([0.00028928, 0.00029001], 174), ([0.00029001, 0.00029075], 173), ([0.00029075, 0.00029148], 172), ([0.00029148, 0.00029222], 171), ([0.00029222, 0.00029295], 170), ([0.00029295, 0.00029369], 169), ([0.00029369, 0.00029443], 168), ([0.00029443, 0.00029516], 167), ([0.00029516, 0.00029931], 166), ([0.00029931, 0.00031944], 165), ([0.00031944, 0.00032018], 164), ([0.00032018, 0.00032091], 163), ([0.00032091, 0.00032165], 162), ([0.00032165, 0.00032238], 161), ([0.00032238, 0.00032312], 160), ([0.00032312, 0.00032386], 159), ([0.00032386, 0.00032459], 158), ([0.00032459, 0.00032533], 157), ([0.00032533, 0.00032606], 156), ([0.00032606, 0.0003268], 155), ([0.0003268, 0.00032754], 154), ([0.00032754, 0.00032827], 153), ([0.00032827, 0.00032901], 152), ([0.00032901, 0.00032974], 151), ([0.00032974, 0.00033048], 150), ([0.00033048, 0.00033122], 149), ([0.00033122, 0.00033195], 148), ([0.00033195, 0.00033269], 147), ([0.00033269, 0.00033342], 146), ([0.00033342, 0.00033416], 145), ([0.00033416, 0.00033489], 144), ([0.00033489, 0.00033563], 143), ([0.00033563, 0.00033637], 142), ([0.00033637, 0.0003371], 141), ([0.0003371, 0.00033784], 140), ([0.00033784, 0.00033857], 139), ([0.00033857, 0.00033931], 138), ([0.00033931, 0.00035255], 137), ([0.00035255, 0.00035329], 136), ([0.00035329, 0.00035402], 135), ([0.00035402, 0.00035476], 134), ([0.00035476, 0.00035549], 133), ([0.00035549, 0.00035623], 132), ([0.00035623, 0.00035697], 131), ([0.00035697, 0.0003577], 130), ([0.0003577, 0.00035844], 129), ([0.00035844, 0.00035917], 128), ([0.00035917, 0.00035991], 127), ([0.00035991, 0.00036065], 126), ([0.00036065, 0.00036138], 125), ([0.00036138, 0.00036212], 124), ([0.00036212, 0.00036285], 123), ([0.00036285, 0.00036359], 122), ([0.00036359, 0.00036432], 121), ([0.00036432, 0.00036506], 120), ([0.00036506, 0.0003658], 119), ([0.0003658, 0.00036653], 118), ([0.00036653, 0.00036727], 117), ([0.00036727, 0.000368], 116), ([0.000368, 0.00036874], 115), ([0.00036874, 0.00036948], 114), ([0.00036948, 0.00037021], 113), ([0.00037021, 0.00037095], 112), ([0.00037095, 0.00037168], 111), ([0.00037168, 0.00037242], 110), ([0.00037242, 0.00037315], 109), ([0.00037315, 0.00037389], 108), ([0.00037389, 0.00037463], 107), ([0.00037463, 0.00037536], 106), ([0.00037536, 0.0003761], 105), ([0.0003761, 0.00037683], 104), ([0.00037683, 0.00037757], 103), ([0.00037757, 0.00037831], 102), ([0.00037831, 0.00039008], 101), ([0.00039008, 0.00039081], 100), ([0.00039081, 0.00039155], 99), ([0.00039155, 0.00039228], 98), ([0.00039228, 0.00039302], 97), ([0.00039302, 0.00039375], 96), ([0.00039375, 0.00039449], 95), ([0.00039449, 0.00039523], 94), ([0.00039523, 0.00039596], 93), ([0.00039596, 0.0003967], 92), ([0.0003967, 0.00039743], 91), ([0.00039743, 0.00039817], 90), ([0.00039817, 0.00039891], 89), ([0.00039891, 0.00039964], 88), ([0.00039964, 0.00040038], 87), ([0.00040038, 0.00040111], 86), ([0.00040111, 0.00040185], 85), ([0.00040185, 0.00040258], 84), ([0.00040258, 0.00040332], 83), ([0.00040332, 0.00040406], 82), ([0.00040406, 0.00040479], 81), ([0.00040479, 0.00040553], 80), ([0.00040553, 0.00040626], 79), ([0.00040626, 0.000407], 78), ([0.000407, 0.00040774], 77), ([0.00040774, 0.00040847], 76), ([0.00040847, 0.00040921], 75), ([0.00040921, 0.00040994], 74), ([0.00040994, 0.00041068], 73), ([0.00041068, 0.00041141], 72), ([0.00041141, 0.00044158], 71), ([0.00044158, 0.00044232], 70), ([0.00044232, 0.00044305], 69), ([0.00044305, 0.00044379], 68), ([0.00044379, 0.00044452], 67), ([0.00044452, 0.00044526], 66), ([0.00044526, 0.000446], 65), ([0.000446, 0.00044673], 64), ([0.00044673, 0.00044747], 63), ([0.00044747, 0.0004482], 62), ([0.0004482, 0.00044894], 61), ([0.00044894, 0.00048131], 60), ([0.00048131, 0.00048205], 59), ([0.00048205, 0.00048278], 58), ([0.00048278, 0.00048352], 57), ([0.00048352, 0.00048426], 56), ([0.00048426, 0.00048499], 55), ([0.00048499, 0.00048573], 54), ([0.00048573, 0.00048646], 53), ([0.00048646, 0.0004872], 52), ([0.0004872, 0.00048794], 51), ([0.00048794, 0.00048867], 50), ([0.00048867, 0.00048941], 49), ([0.00048941, 0.00049014], 48), ([0.00049014, 0.00049088], 47), ([0.00049088, 0.00049161], 46), ([0.00049161, 0.00049235], 45), ([0.00049235, 0.00049309], 44), ([0.00049309, 0.00049382], 43), ([0.00049382, 0.00049456], 42), ([0.00049456, 0.00049529], 41), ([0.00049529, 0.00049603], 40), ([0.00049603, 0.00049677], 39), ([0.00049677, 0.0004975], 38), ([0.0004975, 0.00049824], 37), ([0.00049824, 0.00049897], 36), ([0.00049897, 0.00049971], 35), ([0.00049971, 0.00050045], 34), ([0.00050045, 0.00052988], 33), ([0.00052988, 0.00053061], 32), ([0.00053061, 0.00053135], 31), ([0.00053135, 0.00053208], 30), ([0.00053208, 0.00053282], 29), ([0.00053282, 0.00053355], 28), ([0.00053355, 0.00053429], 27), ([0.00053429, 0.00053503], 26), ([0.00053503, 0.00053576], 25), ([0.00053576, 0.0005365], 24), ([0.0005365, 0.00053723], 23), ([0.00053723, 0.00053797], 22), ([0.00053797, 0.00053871], 21), ([0.00053871, 0.00053944], 20), ([0.00053944, 0.00054018], 19), ([0.00054018, 0.00058285], 18), ([0.00058285, 0.00058359], 17), ([0.00058359, 0.00058432], 16), ([0.00058432, 0.00058506], 15), ([0.00058506, 0.0005858], 14), ([0.0005858, 0.00058653], 13), ([0.00058653, 0.00058727], 12), ([0.00058727, 0.000588], 11), ([0.000588, 0.00058874], 10), ([0.00058874, 0.00063877], 9), ([0.00063877, 0.00063951], 8), ([0.00063951, 0.00064025], 7), ([0.00064025, 0.00064098], 6), ([0.00064098, 0.00064172], 5), ([0.00064172, 0.00069543], 4), ([0.00069543, 0.00069617], 3), ([0.00069617, 0.0006969], 2), ([0.0006969, 0.00069764], 1)]
        # self.draw_mode(from_octave_reg_2, title='Мода после регуляризации канал 2')

if __name__ == "__main__":
    data_set = '700nm_0.23mm.csv'
    files = [f'./data/Канал 1_{data_set}', f'./data/Канал 2_{data_set}']
    analysis = SolarAnalysis(data_set, files)
    # save_p_first = f'./results/first_lab'
    # analysis.first_lab(save_p_first)
    # analysis.prepare_data_for_matlab('./octave/data')
    # analysis.second_lab(f'./results/second_lab')
    analysis.third_lab(f'./results/third_lab')
