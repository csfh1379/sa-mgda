% M_phi(x,y) = (psi(x,y)-y, psi(y,x)+x)
% psi(x,y) = 1/8*x*(-1 + x^2 + y^2)*(-1 + 4*x^2 + 4*y^2)

global L lb ub
L = sqrt(5449798437-173756*sqrt(890712929))/10000;
lb = -11/10; ub = 11/10;
J=10;
tau = 1/(2*L);
num_iter = 1e3;

x_1 = 1; y_1 = 1/2;

%% MGDA
Lips_reg_list = [0, 1/4*L, L];
for i = 1:3
    Lips_reg = Lips_reg_list(i);
    Lips_max_func_x = 2*L; % An aribitrary step size.
    Lips_func_y = L + Lips_reg;
    gd_norm_sq = zeros(1,num_iter);
    x = cell(1,num_iter); y = cell(1,num_iter);
    x{1} = x_1; y{1} = y_1;
    for k = 1:num_iter
        x{k+1} = clip( x{k} - 1/Lips_max_func_x*gd_x(x{k},y{k}) );
        y{k+1} = y{k};
        for j = 1:J
            y{k+1} = clip( y{k+1} + 1/Lips_func_y*(gd_y(x{k+1},y{k+1}) - Lips_reg*(y{k+1}-y_1)) );
        end
        gd_norm_sq(k) = gd_norm(x{k+1},y{k+1});
    end
    gd_norm_sq_best = best(gd_norm_sq);
    data_mgda{i} =  gd_norm_sq;
    data_mgda_best{i} = gd_norm_sq_best;
end


%% SA-MGDA
Lips_func_y = 2*L + 1/tau; 
gd_norm_sq = zeros(1,num_iter);
x = cell(1,num_iter); y = cell(1,num_iter);
x{1} = x_1; y{1} = y_1;
for k = 1:num_iter
    x{k+1} = clip( x{k} - tau*gd_x(x{k},y{k}) );
    y{k+1} = y{k};
    for j = 1:J
        y{k+1} = clip( 1/(1+2*L*tau)*(y{k} - tau*gd_y(x{k},y{k})) ...
            + 2*L*tau/(1+2*L*tau)*(y{k+1} + 1/L*gd_y(x{k+1},y{k+1})) );
    end
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
plot(iter,data_mgda{1},'-o','MarkerIndices',50,'Linewidth',linewidth)
plot(iter,data_mgda{2},'-^','MarkerIndices',50,'Linewidth',linewidth)
plot(iter,data_mgda{3},'-s','MarkerIndices',50,'Linewidth',linewidth)
plot(iter,data_sa_mgda,'-x','MarkerIndices',50,'Linewidth',linewidth)
legend('MGDA, $\lambda = 0$','MGDA, $\lambda = L/4$','MGDA, $\lambda = L$','SA-MGDA','Location','southeast','interpreter','latex')
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
ylim([1e-10,2])
ylabel('$\|Mx_k\|^2$','interpreter','latex')
xlabel('Iterations')
set(gca,'FontSize',15)
hold off

%% Functions

function val = clip(x)
    global lb ub
    val = max(lb,min(ub,x));
end

function vec = best(x)
    vec = zeros(1,length(x));
    for i = 1:length(x)
        vec(i) = min(x(1:i));
    end
end

function val = psi(x,y)
    val = 1/8*x*(-1+x^2+y^2)*(-1+4*x^2+4*y^2);
end

function grad = gd_x(x,y)
    grad = psi(x,y)-y;
end

function grad = gd_y(x,y)
    grad = -psi(y,x)-x;
end

function value = gd_norm(x,y)
    value = norm(gd_x(x,y))^2 + norm(gd_y(x,y))^2;
end