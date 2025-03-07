function [figname] = E_plot_slice_arbitrary_riehen()

datadir = '../../data-riehen/run3_dcV2_mul2_g200m'
dirinv = [datadir '/vs-model/run0_dv60_dz50m_N100_10L']
fname = [dirinv '/combined_result.mat']
load(fname, 'Vs_min_mat', 'Vs_sol_mat', 'x_profile', 'y_profile', 'z_grid_eff')
z_grid_eff = z_grid_eff .*1e-3; % m to km

load("../../data-swisstopo/swisstopo-mat/deep_wells_RiehenGrid.mat", 'deepwells')
swisstopodir = '../../data-swisstopo/swisstopo-mat/Accident_tecto_riehen/';

output_fold = [dirinv '/plots']
if ~isfolder([output_fold])
    mkdir([output_fold])
end

load([datadir, '/grid/map_matrix_terrain.mat'], 'map', 'x_map', 'y_map')
load([datadir, '/grid/kernel.mat'], 'x_stat', 'y_stat')

%% Plot params
dist_tol = 0.5  % tolerance for nearby stations/faults for depth sections
dist_tol_f = 0.5 
zmax = 5 %3.5
colorlims = nan %[1500 3500]
amplitude_type = 'subtract_constant' % absolute, subtract_1D, subtract_constant
% amplitude_type = 'absolute'
contourcolor = [1 1 1].*0.5;
contourwidth = 0.5;
contourstyle = ':';
labelfontsize = 12;
levels = nan; %[-50:10:-10 -5:1:5 10:10:50];

%% Define slices
load([datadir '/dist_stat.mat'], 'SW_corner')

% Guerler et al. 1984 P5
% pointll2 = [47.556074, 7.701086]; % East
% pointll1 = [47.599152, 7.555517]; % West - Deviated to be straight line
% pointll1 = [47.584044, 7.548651]; % West as in true P5
% pointll2 = [47.581265, 7.617573]; % break point P5
% figname = [output_fold '/slice_guerler_prof5_' amplitude_type '.png']

% P1
% pointll1 = [47.542407, 7.673953]; % S
% pointll2 = [47.644947, 7.698887]; % N
% figname = [output_fold '/slice_guerler_prof1_' amplitude_type '.png']

% P3
pointll1 = [47.610536, 7.577393];
pointll2 = [47.616351, 7.706536];
figname = [output_fold '/slice_guerler_prof3_' amplitude_type '.png']

% Convert start-end points lat,long to x,y
[x1,y1] = ll2xy(pointll1(1),pointll1(2), SW_corner);
[x2,y2] = ll2xy(pointll2(1),pointll2(2), SW_corner);
pointxy1 = [x1,y1];
pointxy2 = [x2,y2];

slice_length = sqrt((pointxy1(1)-pointxy2(1))^2+(pointxy1(2)-pointxy2(2))^2);


%% Colormaps
cmap_pm = cbrewer2('div','RdYlBu',50,'linear');
cmap_abs = flipud(jet(50));

%% Index without data
ind_nodata = find(sum(Vs_sol_mat,3)==0);
Vs_sol_mat(ind_nodata) = nan;

% From G_mat
load([datadir, '/grid/kernel.mat'], 'G_mat','x_grid','y_grid')
thres_dist = 0.01; % km
min_density = 1;
G3D = reshape(G_mat',[length(x_grid) length(y_grid) size(G_mat',2)]);
G_count = zeros(size(G3D));
ind_G_ray = G3D(:) > thres_dist; % count ray if >10m in cell
G_count(ind_G_ray) = 1;
G_sum = sum(G_count,3);
mask2D = nan(size(G_sum));
mask2D(G_sum >= min_density) = 1;
ind_no_data = find(G_sum < min_density);
mask3D = repmat(mask2D, [1 1 size(Vs_sol_mat,3)]);
Vs_sol_mat = Vs_sol_mat .* mask3D;

%% Mean Vs 1D profile
mean_profile = zeros([length(z_grid_eff), 1]);
count = 0;
for ix=1:length(x_grid)
    for iy=1:length(y_grid)
        if G_sum(ix,iy) > 3 %~isnan(mask2D(ix,iy))
            mean_profile = mean_profile + squeeze(Vs_sol_mat(ix,iy,:));
            count = count + 1;
        end
    end
end
mean_profile = mean_profile./count;

%% smooth Vs cube
nx_smooth = 3; 
ny_smooth = 3;
nz_smooth = 3;
Vs_smooth = smooth3(Vs_sol_mat,'box',[nx_smooth ny_smooth nz_smooth]);
% Vs_smooth=smooth3(Vs_min_mat,'box',[nx_smooth ny_smooth nz_smooth]);
% Vs_smooth = Vs_sol_mat;
Vs_smooth(ind_nodata) = nan;

%% Interpolate 3D Vs for visualization
% Vs_smooth, x_profile, y_profile, z_grid_eff
dx_new = 0.05; % km
dy_new = 0.1; % km
x_profile_new = x_profile(1):dx_new:x_profile(end);
y_profile_new = y_profile(1):dy_new:y_profile(end);
[Xq,Yq,Zq] = ndgrid(x_profile_new, y_profile_new, z_grid_eff);
Vs_smooth_new = interpn(x_profile, y_profile, z_grid_eff, Vs_smooth, Xq,Yq,Zq);

%% Define the plane of the slice
pointA = [pointxy1 0]; 
pointB = [pointxy1 zmax]; 
pointC = [pointxy2 0]; % 3 points contained by the plane
normal = cross(pointA - pointB, pointA - pointC); % Calculate plane normal
% Transform points to x,y,z
x = [pointA(1) pointB(1) pointC(1)];  
y = [pointA(2) pointB(2) pointC(2)];
z = [pointA(3) pointB(3) pointC(3)];
% Find all coefficients of plane equation    
A = normal(1); B = normal(2); C = normal(3);
D = -dot(normal,pointA);

% Decide on a suitable showing range/ grid spacing
xx = x_profile_new;
zz = z_grid_eff;
[xgridsl,zgridsl] = meshgrid(xx,zz);
ygridsl = (A * xgridsl + C * zgridsl + D) / (-B);
xslice = xgridsl(1,:); 
yslice = ygridsl(1,:);
distslice0 = sqrt((xgridsl-pointxy1(1)).^2 + (ygridsl-pointxy1(2)).^2);
distslice = distslice0(1,:);

% figure();clf
% surf(xgridsl,ygridsl,zgridsl);alpha(0.3);shading flat
% hold on
% [XGRID,YGRID,ZGRID] = ndgrid(x_grid,y_grid,z_grid_eff(1):500:z_grid_eff(end));
% plot3(XGRID(:),YGRID(:),ZGRID(:),'k+')
% grid on;

%% Interpolate on the plane's grid points
Vs_slice = interpn(x_profile_new,y_profile_new,z_grid_eff,Vs_smooth_new,xgridsl,ygridsl,zgridsl); 
% Cut to length of slice
[~,indsl1] = min(distslice);
indsl2 = find(distslice <= slice_length,1,'last');
indsl = indsl1:indsl2;
distslice = distslice(indsl);
Vs_slice = Vs_slice(:,indsl);

%% Plot
figure(1);clf;
set(gcf,'color','w');

% pcolor(distf,zz,Vs_slice); shading flat; axis xy
hold on
depth = z_grid_eff;
switch amplitude_type
    case 'subtract_constant'
        mean_Vs = nanmean(Vs_smooth(:));
        Vs_slice_ref = ones(size(Vs_slice)) .* mean_Vs;
        Vs_slice_anom = (Vs_slice - Vs_slice_ref) ./ Vs_slice_ref * 100;
        im = pcolor(distslice,depth,Vs_slice_anom);
%         set(im, 'facealpha','flat','alphadata', mask')
        shading('interp');
        colormap(cmap_pm);
        hb=colorbar;
        ylabel(hb,'Vs anomaly (%)','fontsize',labelfontsize)
        max_anom = nanmax(abs(Vs_slice_anom(:)));
        if isfinite(max_anom) && max_anom > 0
            set(gca, 'CLim', [-max_anom, max_anom]);
        end
        hold on;
        if ~isnan(levels)
            hcont = contour(distslice,depth,Vs_slice_anom,levels,'color',contourcolor,'LineWidth',contourwidth,'LineStyle',contourstyle);
        else
            [M,c] = contour(distslice,depth,Vs_slice_anom,'color',contourcolor,'LineWidth',contourwidth,'LineStyle',contourstyle);
        end
    case 'absolute'
        im = pcolor(distslice,depth,Vs_slice);
%         set(im, 'facealpha','flat','alphadata', mask')
        if ~isnan(levels)
            contour(distslice,depth,Vs_slice,levels,'color',contourcolor,'LineWidth',contourwidth,'LineStyle',contourstyle);
        else
            contour(distslice,depth,Vs_slice,'color',contourcolor,'LineWidth',contourwidth,'LineStyle',contourstyle);
        end
        shading('interp');
        colormap(cmap_abs);
        hb=colorbar;
        ylabel(hb,'Vs (m/s)','fontsize',labelfontsize)
end

% Add intersecting faults
plot_faults(pointxy1, pointxy2)
% Add wells if close
add_wells(pointxy1,pointxy2,pointll1,pointll2,xslice,yslice,deepwells)

% axis equal
axis tight
axis equal
box on
set(gca,'ydir','reverse','fontsize',12,'TickDir','out')
xlabel('Distance along slice')
ylabel('Depth (km)')
xlim([0, slice_length])
ylim([0 zmax])

export_fig(figname)

end

function [] = add_wells(pointxy1,pointxy2,pointll1,pointll2,xslice,yslice,deepwells)

    dist_tol = 1.5;
    markerfacecolor = 'r';
    marker = 'hexagram';
    markeredgecolor = 'k';
    linewidth = 2;  
    min_depth = 800;  
    markersize = 80;    
    ind = find(deepwells.latitude > min([pointll1(1),pointll2(1)])-0.1 & ...
        deepwells.latitude < max([pointll1(1),pointll2(1)])+0.1 & ...
        deepwells.longitude > min([pointll1(2),pointll2(2)])-0.1 & ...
        deepwells.longitude < max([pointll1(2),pointll2(2)])+0.1 & ...
        deepwells.depth >= min_depth);
    deepwells = deepwells(ind,:)
    for k=1:size(deepwells,1)
        deepwells(k,:).name
        [mindist, Imin] = min(sqrt( (xslice - deepwells(k,:).xgrid).^2 + (yslice - deepwells(k,:).ygrid).^2));
        if mindist < dist_tol
            distwell = sqrt((xslice(Imin)-pointxy1(1)).^2 + (yslice(Imin)-pointxy1(2)).^2);
            plot([distwell,distwell], [0, deepwells(k,:).depth*1e-3],LineStyle="--",LineWidth=1,color='k')
            scatter(distwell, 0, markersize, markerfacecolor, 'filled', marker, 'MarkerEdgeColor', markeredgecolor, 'LineWidth',linewidth);
%             text(distwell,0.1, deepwells(k,:).name)
        end
    end
end

function [] = plot_faults(pointxy1, pointxy2)

%     hcur = gcf;

    swisstopodir_faults = '../../data-swisstopo/swisstopo-mat/Accident_tecto_riehen/';

    symbol = '|k';
    linewidth = 3;
    markersize = 8;
    markerfacecolor = 'k';
    
    flist = dir(fullfile(swisstopodir_faults,'*'));
    flist = flist(3:end); % Remove '.' and '..'
%     figure(10); clf; h10 = gcf; hold on
    for k=1:length(flist)
    
        file = flist(k).name;
        load(fullfile(swisstopodir_faults,file))
%         figure(h10)
%         plot(x,y,'k-')
%         plot([pointxy1(1),pointxy2(1)],[pointxy1(2),pointxy2(2)],'ro')
        % Check if fault intersects
        Lfault = [x;y];
        Lslice = [pointxy1(1),pointxy2(1);pointxy1(2),pointxy2(2)];
        P = InterX(Lfault,Lslice);
        if ~isempty(P)
            P;
            distcross = sqrt((P(1)-pointxy1(1)).^2 + (P(2)-pointxy1(2)).^2);
            
%             figure(hcur)
            plot(distcross,0,symbol, 'MarkerSize',markersize, 'MarkerFaceColor',markerfacecolor,'LineWidth',linewidth)
            

%             figure(10); hold on
%             plot(x,y,'r-')

%             figure(hcur)
        end
    end
%     figure(hcur)
end
%%
function [x,y] = ll2xy(lat,long, SW_corner)
% Convert lat,long to X,Y using spherical coord projection
R_earth = 6371;
% SW_corner = [min_lat min_lon]
ref_lat_glob = SW_corner(1) * ones(size(lat));  % south west corner chosen as grid origin
ref_lon_glob = SW_corner(2) * ones(size(long));
x = R_earth * cos((lat + ref_lat_glob)/2*pi/180).*(long - ref_lon_glob) *pi/180;
y = R_earth * (lat - ref_lat_glob) * pi/180;

end

%%
function add_slice_arbitrary(pointll1, pointll2, maxdepth)

% Convert start-end points lat,long to x,y
[x1,y1] = ll2xy(pointll1(1),pointll1(2), ref_lat_glob, ref_lon_glob);
[x2,y2] = ll2xy(pointll2(1),pointll2(2), ref_lat_glob, ref_lon_glob);
pointxy1 = [x1,y1];
pointxy2 = [x2,y2];

% Define the plane of the slice
pointA = [pointxy1 0]; 
pointB = [pointxy1 maxdepth]; 
pointC = [pointxy2 0]; % 3 points contained by the plane
normal = cross(pointA - pointB, pointA - pointC); % Calculate plane normal
% Transform points to x,y,z
x = [pointA(1) pointB(1) pointC(1)];  
y = [pointA(2) pointB(2) pointC(2)];
z = [pointA(3) pointB(3) pointC(3)];
% Find all coefficients of plane equation    
A = normal(1); B = normal(2); C = normal(3);
D = -dot(normal,pointA);

% Decide on a suitable showing range/ grid spacing
xx = min(x):1:max(x);
zz = min(z):1:max(z);
[xgridsl,zgridsl] = meshgrid(xx,zz);
ygridsl = (A * xgridsl + C * zgridsl + D) / (-B);
distslice = sqrt((xgridsl-pointxy1(1)).^2 + (ygridsl-pointxy1(2)).^2);
distf = distslice(1,:);

% Interpolate on the plane's grid points

Vsinterp = interp3(xgrid,ygrid,zgrid,Vs,xgridsl,ygridsl,zgridsl); 
% Vsinterp(dwsinterp<thresh) = NaN;

end