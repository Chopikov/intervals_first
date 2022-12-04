pkg load interval
addpath(genpath('./octave-interval-examples/m'))
addpath(genpath('./m'))
addpath(genpath('./data'))


load("data_2.mat")
x = data_2(:, 1);
y = data_2(:, 2);

%draw_graph(x, y, "n", "mV", "", 1, "all_data.eps");
%print -djpg all_data.jpg
%draw_graph(x, y, "time", "value", "", 1, "filtered_data.eps");
%draw_graph(x, y, "time", "value", "d", 1, "selected_data.eps");

%draw_graph(x, y, "n", "mV", "", 1, "all_data.eps");
%print -djpg all_data.jpg
%draw_graph(x, y, "time", "value", "", 1, "filtered_data.eps");
%draw_graph(x, y, "time", "value", "d", 1, "selected_data.eps");

%dot_problem(x, y);

##irp_temp = interval_problem(x, y);
##[b_maxdiag, b_gravity] = parameters(x, y, irp_temp);
##joint_depth(irp_temp, b_maxdiag, b_gravity);
##prediction(x, y, irp_temp, b_maxdiag, b_gravity);
##edje_points(x, y, irp_temp);

edge_points = [14 167];

x_1 = x(1:edge_points(1) - 1);
y_1 = y(1:edge_points(1) - 1);
dot_problem(x_1, y_1);

irp_temp = interval_problem(x_1, y_1);
[b_maxdiag, b_gravity] = parameters(x_1, y_1, irp_temp);
joint_depth(irp_temp, b_maxdiag, b_gravity);

x_2 = x(edge_points(1):edge_points(2) - 1);
y_2 = y(edge_points(1):edge_points(2) - 1);
dot_problem(x_2, y_2);

irp_temp = interval_problem(x_2, y_2);
[b_maxdiag, b_gravity] = parameters(x_2, y_2, irp_temp);
joint_depth(irp_temp, b_maxdiag, b_gravity);

x_3 = x(edge_points(2):200);
y_3 = y(edge_points(2):200);
dot_problem(x_3, y_3);

irp_temp = interval_problem(x_3, y_3);
[b_maxdiag, b_gravity] = parameters(x_3, y_3, irp_temp);
joint_depth(irp_temp, b_maxdiag, b_gravity);
