% This is the code for drawing the prediction error of case 1 & 2


load('case1_n2000.mat')
% load('case1_n5000.mat')
% load('case2_n2000.mat')
% load('case2_n5000.mat')

err_data = [err_hat_vcplt(use_flag, :), err_hat_ptt(use_flag, :), err_hat_krus(use_flag, :)];

model_label = repelem({'M1', 'M2', 'M3'}, 4);
At_label = repmat({'A1(t)', 'A2(t)', 'A3(t)', 'A4(t)'}, 1, 3);

err_data_vec = err_data(:)';
group_model_label = repelem(model_label, sum(use_flag));
group_At_label = repmat(At_label, 1, sum(use_flag));

figure;
colors = lines(3);
boxplot(err_data_vec, {group_At_label, group_model_label}, 'factorgap', 5, 'colorgroup', group_model_label, 'colors', colors)

xticklabels = {'A1(t)', 'A2(t)', 'A3(t)', 'A4(t)'};
xticks = [2, 5.55, 9.08, 12.65];
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);

% title('Case II, n=5000')
xlabel('A(t)')
ylabel('Prediction error')

hold on;
h = findobj(gca, 'Tag', 'Box');
legend_labels = {'M1', 'M2', 'M3'};

color_m1 = colors(1, :);
color_m2 = colors(2, :);
color_m3 = colors(3, :);

h_legend = zeros(3, 1);
h_legend(1) = plot(NaN, NaN, 's', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', colors(1, :));
h_legend(2) = plot(NaN, NaN, 's', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', colors(2, :));
h_legend(3) = plot(NaN, NaN, 's', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', colors(3, :));

legend(h_legend, legend_labels, 'Location', 'northeast');
hold off;

