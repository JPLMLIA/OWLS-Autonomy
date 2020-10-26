% part of the ACME hand label generation Pipeline
% make csv out of labels that were exported as table from Matlab's 'Image Labeler'
%
% Steffen Mauceri
% updated: Oct 2020



name = '190604012.csv';
Label = gTruth;

time_px = [];
mass_row = [];
ambigious_flag = [];
scale_0_4100 = [];

for row = 1:height(Label)
   %iterate over peaks
   if ~isempty(Label.peak{row, 1})
       n_peaks = length(Label.peak{row, 1});
       for n = 1:n_peaks
           % get first peak
           peak_n = Label.peak{row, 1}(n).Position;
           % get time px
           time_px = [time_px, round(peak_n(1) + 0.5 * peak_n(3))];
           mass_idx = Label.imageFilename{row,1}; 
           mass_idx = split(mass_idx, '_'); mass_idx = mass_idx{end};
           mass_idx = split(mass_idx, '.'); mass_idx = str2num(mass_idx{1});
           mass_row = [mass_row, mass_idx];
           ambigious_flag = [ambigious_flag, 0];
       end
   end
   
   % iterate over ambigious peaks
   if ~isempty(Label.ambigious_peak{row, 1})
       n_peaks = length(Label.ambigious_peak{row, 1});
       for n = 1:n_peaks
           % get first peak
           peak_n = Label.ambigious_peak{row, 1}(n).Position;
           % get time px
           time_px = [time_px, round(peak_n(1) + 0.5 * peak_n(3))];
           mass_idx = Label.imageFilename{row,1}; 
           mass_idx = split(mass_idx, '_'); mass_idx = mass_idx{end};
           mass_idx = split(mass_idx, '.'); mass_idx = str2num(mass_idx{1});
           mass_row = [mass_row, mass_idx];
           ambigious_flag = [ambigious_flag, 1];
       end
   end
   
   % get convertion from time_px to time_idx
   if ~isempty(Label.time_idx_0_to_4100{row,1})
       scale = Label.time_idx_0_to_4100{row,1}.Position;
       scale_0_4100 = [scale_0_4100; [scale(1), scale(1) + scale(3)]];
   end
   
end

%% convert to mass_idx and time_idx
mass_idx = mass_row;% * 7 - 3; %4 11 18

% calculate convertion from time_px to time_idx
scale_0_4100 = mean(scale_0_4100,1);
time_idx = round(interp1(scale_0_4100,[0,4100], time_px, 'lin', 'extrap'));

% make a plot of labels
scatter(time_idx(ambigious_flag == 1), mass_idx(ambigious_flag == 1))
hold on
scatter(time_idx(ambigious_flag == 0), mass_idx(ambigious_flag == 0))
legend('Ambigious Peaks', 'Peak')
hold off

%%  write csv
output = [mass_idx',time_idx',ambigious_flag'];
output = array2table(output);
output.Properties.VariableNames = {'mass_idx', 'time_idx','ambigious_flag'};
writetable(output, name)

