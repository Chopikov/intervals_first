import os

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.linear_model import LinearRegression


class SolarAnalysis:
    def __init__(self, data_set, save_path, files):
        self.data_set = data_set
        self.save_path = save_path
        self.need_save = True
        self.data_list = self.read_data(files)
        self.eps = 1.5 * 1e-4

    def read_data(self, files_with_data):
        data_rows = []
        for file in files_with_data:
            data_r = genfromtxt(file, delimiter=';', encoding='cp1251')
            data_rows.append([val[0] for val in data_r][1:201])
        return data_rows

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

    def get_out_coef(self):
        out_coef = []
        for i in range(len(self.data_list[0])):
            out_coef.append(self.data_list[0][i] / self.data_list[1][i])
        return out_coef

    def save_and_show(self, file_name):
        if self.need_save:
            plt.savefig(f'{self.save_path}/{file_name}')
        plt.show()

    def plt_errorbar(self, x, data, yerr, title, label=''):
        plt.errorbar(x, data, yerr=yerr, marker='.', linestyle='none', ecolor='g', elinewidth=0.5, capsize=2,
                     capthick=1, label=label)
        plt.xlabel("n")
        plt.ylabel("мВ")
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

    def get_inner_r(self, lin_data, w_list, R_out):
        step_count = 1000
        r_step = (R_out[1] - R_out[0]) / step_count
        start = R_out[0]
        jaccar_list = []
        while start <= R_out[1]:
            data_l = [[data_item - self.eps * w_list[0][num], data_item + self.eps * w_list[0][num]]
                      for num, data_item in enumerate(list(lin_data[0]))]
            data_l += [[start * (data_item - self.eps * w_list[1][num]),
                        start * (data_item + self.eps * w_list[1][num])]
                       for num, data_item in enumerate(list(lin_data[1]))]
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

    def draw_all_intervals(self, lin_data, w_list, optimal_m):
        data_len = (len(lin_data[0]))
        x = np.arange(1, data_len + 1, 1, dtype=int)
        yerr = [self.eps] * data_len
        for i in range(data_len):
            yerr[i] *= w_list[0][i]
        plt.errorbar(x, lin_data[0], yerr=yerr, ecolor='blue', label='Канал 1')
        yerr = [self.eps] * data_len
        for i in range(data_len):
            yerr[i] *= (w_list[1][i] * optimal_m)
            lin_data[1][i] *= optimal_m
        plt.errorbar(x, lin_data[1], yerr=yerr, ecolor='yellow', label='Канал 2')
        plt.legend(frameon=False)
        plt.title(f'Пересечение интервалов')
        plt.xlabel("n")
        plt.ylabel("мВ")
        self.save_and_show('intersection.png')

    def main(self):
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
        opt_m = self.get_inner_r(lin_data, w_l, R_outer)
        print(f"Ropt = {opt_m}")
        if opt_m is not None:
            self.draw_all_intervals(lin_data, w_l, opt_m[0])


if __name__ == "__main__":
    data_set = '700nm_0.23mm.csv'
    save_p = f'./results'
    files = [f'./data/Канал 1_{data_set}', f'./data/Канал 2_{data_set}']
    analysis = SolarAnalysis(data_set, save_p, files)
    analysis.main()
