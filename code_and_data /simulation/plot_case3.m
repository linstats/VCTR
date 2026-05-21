clear all

load('case3_pred_err.mat')

err_data = zeros(length(use_rows), 5,3);
err_data(:,1,:) = err_hat_gt(use_rows,:);
err_data(:,2,:) = err_hat_sm(use_rows,:);
err_data(:,3,:) = err_hat_lasso(use_rows,:);
err_data(:,4,:) = err_hat_scad(use_rows,:);
err_data(:,5,:) = err_hat_mcp(use_rows,:);
err_data = reshape(err_data, length(use_rows), []);

rho_label = repelem({'0.1', '0.5', '0.9'}, 5);
pen_label = repmat({'oracle', 'sm', 'Lasso', 'SCAD', 'MCP'}, 1, 3);

err_data_vec = err_data(:)';
group_rho_label = repelem(rho_label, 1, length(use_rows));
group_pen_label = repelem(pen_label, 1, length(use_rows));

figure;
colors = lines(5);
boxplot(err_data_vec, {group_rho_label, group_pen_label}, 'factorgap', 5, 'colorgroup', group_pen_label, 'colors', colors)

xticklabels = {'0.1', '0.5', '0.9'};
xticks = [3, 8.7, 14.4];
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);

% title('Case III, n=5000')
xlabel('Covariance parameter \rho')
ylabel('Prediction error')

hold on;
h = findobj(gca, 'Tag', 'Box');
legend_labels = {'Oracle', 'Unpenalized', 'Lasso', 'SCAD', 'MCP'};

color_oracle = colors(1, :);
color_sm = colors(2, :);
color_lasso = colors(3, :);
color_scad = colors(4, :);
color_mcp = colors(5, :);

h_legend = zeros(5, 1);
h_legend(1) = plot(NaN, NaN, 's', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', colors(1, :));
h_legend(2) = plot(NaN, NaN, 's', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', colors(2, :));
h_legend(3) = plot(NaN, NaN, 's', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', colors(3, :));
h_legend(4) = plot(NaN, NaN, 's', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', colors(4, :));
h_legend(5) = plot(NaN, NaN, 's', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', colors(5, :));

legend(h_legend, legend_labels, 'Location', 'northeast');
hold off;


