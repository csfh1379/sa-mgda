% phi(x,y) = a/2 * x^2 + b*x*y - b/2 * y^2

global L a b
L = 1; rho = -1/4/L;
a = rho/2*L^2; b = L*sqrt(1 - rho^2/4*L^2);

num_iter = 1e3;     % Number of iterations
x_1 = 0; y_1 = 1;   % Initial point

%% MGDA
Lips_reg_list = [L, L/2, L/4];  % List of regularization parameters
for i = 1:3
    Lips_reg = Lips_reg_list(i);
    Lips_max_func = abs(a) + b^2/(Lips_reg + a);
    gd_norm_sq = zeros(1,num_iter);
    x = cell(1,num_iter); y = cell(1,num_iter);
    x{1} = x_1; y{1} = y_1;
    for k = 1:num_iter
        x{k+1} = x{k} - 1/Lips_max_func*gd_x(x{k},y{k});
        y{k+1} = (b*x{k+1} + Lips_reg*y_1)/(a+Lips_reg);
        gd_norm_sq(k) = gd_norm(x{k+1},y{k+1});
    end
    gd_norm_sq_best = best(gd_norm_sq);
    data_mgda{i} =  gd_norm_sq;
    data_mgda_best{i} = gd_norm_sq_best;
end


%% SA-MGDA
gd_norm_sq = zeros(1,num_iter);
x = cell(1,num_iter); y = cell(1,num_iter);
x{1} = x_1; y{1} = y_1;
for k = 1:num_iter
    x{k+1} = x{k} - 1/L*gd_x(x{k},y{k});
    y{k+1} = (y{k} + 2/L*b*x{k+1} - 1/L*gd_y(x{k},y{k}))/(1+2/L*a);
    gd_norm_sq(k) = gd_norm(x{k+1},y{k+1});
end
gd_norm_sq_best = best(gd_norm_sq);
data_sa_mgda = gd_norm_sq;
data_sa_mgda_best = gd_norm_sq_best;




%% Plotting
color = {[0 0.4470 0.7410],[0.8500 0.3250 0.0980],[0.9290 0.6940 0.1250],...
    [0.4940, 0.1840, 0.5560],[0.4660, 0.6740, 0.1880]};
linewidth=2;
figure
hold on
iter = 1:num_iter;
plot(iter,data_mgda{1},'-o','MarkerIndices',20,'Linewidth',linewidth)
plot(iter,data_mgda{2},'-^','MarkerIndices',20,'Linewidth',linewidth)
plot(iter,data_mgda{3},'-s','MarkerIndices',20,'Linewidth',linewidth)
plot(iter,data_sa_mgda,'-x','MarkerIndices',20,'Linewidth',linewidth)
legend('MGDA, $\lambda = 1$','MGDA, $\lambda = 1/2$','MGDA, $\lambda = 1/4$','SA-MGDA','Location','southeast','interpreter','latex')
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
ylim([1e-20,10])
xlim([1,1e3])
ylabel('$\|Mx_k\|^2$','interpreter','latex')
xlabel('Iterations')
set(gca,'FontSize',15)
hold off

%% Functions
function vec = best(x)
    vec = zeros(1,length(x));
    for i = 1:length(x)
        vec(i) = min(x(1:i));
    end
end

function grad = gd_x(x,y)
    global a b
    grad = a*x + b*y;
end

function grad = gd_y(x,y)
    global a b
    grad = b*x - a*y;
end

function value = gd_norm(x,y)
    value = norm(gd_x(x,y))^2 + norm(gd_y(x,y))^2;
end