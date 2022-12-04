function edje_points(x, y, irp_temp)
	m = length(x);
	X = [ x.^0 x ];
	# Поиск граничных точек
  MY_EPS = 1e-7;
	## Значения y, предсказанные с помощью модели y = beta1 + beta2 * x в точках эксперимента
	cur_x = irp_temp.y;
	cur_eps = irp_temp.epsilon;

	yp0 = ir_predict(irp_temp, X);
  vec_of_edje_points = []

	for i = 1:m
		x_top = cur_x(i) + cur_eps(i);
		x_bot = cur_x(i) - cur_eps(i);

		y_top = yp0(i, 2);
		y_bot = yp0(i, 1);
		%fprintf("i: %d, top: %.3f, bottom: %.3f\n", i, abs(y_top - x_top), abs(y_bot - x_bot));
		if abs(y_top - x_top) < MY_EPS
			display(i);
      vec_of_edje_points = [vec_of_edje_points i];
		endif
		if abs(y_bot - x_bot) < MY_EPS
			display(i);
      vec_of_edje_points = [vec_of_edje_points i];
		endif
	endfor

	border_x = [-0.1, 0.1];
	border_y = [-5e-4, 5e-4];

%	for i = 1:m
  for i = vec_of_edje_points
		cur_point_x = x(i);
		cur_point_y = y(i);
		figure('position',[0, 0, 800, 600]);
    xlimits = [1 220];
		ir_plotmodelset(irp_temp, xlimits);     # коридор совместных зависимостей
		hold on;
		ir_scatter(irp_temp,'bo');              # интервальные измерения
		grid on;
		set(gca, 'fontsize', 12);
		xlim([cur_point_x + border_x(1), cur_point_x + border_x(2)]);
		ylim([cur_point_y + border_y(1), cur_point_y + border_y(2)]);
		%saveas(gcf, sprintf("../report/images/edge_point_%d.eps", i), "epsc");
	endfor
endfunction
