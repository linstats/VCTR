clear all

load('case4_pred_err.mat');

err_data_vec = [err_hat_gt{1,1}', err_hat_gt{2,1}', err_hat_lasso{1,1}', err_hat_lasso{2,1}', err_hat_scad{1,1}', err_hat_scad{2,1}', err_hat_mcp{1,1}', err_hat_mcp{2,1}'];

n_label = repmat([repelem({'n=2000'}, 1,2000), repelem({'n=5000'}, 1,5000)],1,4);
pen_label = repelem({'Oracle', 'Lasso', 'SCAD', 'MCP'}, 1, 2000+5000);

figure;
colors = lines(2);
boxplot(err_data_vec, {pen_label, n_label}, 'factorgap', 5, 'colorgroup', n_label, 'colors', colors)

xticklabels = {'Oracle', 'Lasso', 'SCAD', 'MCP'};
xticks = [1.5, 3.85, 6.2, 8.55];
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);

% title('Case IV')
xlabel('Penalty type')
ylabel('Prediction error')

hold on;
h = findobj(gca, 'Tag', 'Box');
legend_labels = {'n=2000', 'n=5000'};

color_n2000 = colors(1, :);
color_n5000 = colors(2, :);

h_legend = zeros(2, 1);
h_legend(1) = plot(NaN, NaN, 's', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', colors(1, :));
h_legend(2) = plot(NaN, NaN, 's', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', colors(2, :));

legend(h_legend, legend_labels, 'Location', 'northeast');
hold off;

