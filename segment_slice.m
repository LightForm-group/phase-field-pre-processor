function [grainIDs] = segment_slice(argsJSONPath)

    close all;

    allArgs = jsondecode(fileread(argsJSONPath));

    SS = {specimenSymmetry(allArgs.specimen_symmetry)};
    CS = cell(length(allArgs.crystal_symmetries));
    phaseNames = cell(length(allArgs.crystal_symmetries));

    for CSIdx = 1:length(allArgs.crystal_symmetries)

        CSData = allArgs.crystal_symmetries(CSIdx);
        phaseNames{CSIdx} = CSData.mineral;

        alignment = {};

        if isfield(CSData.unit_cell_alignment, 'x')
            alignment{end + 1} = sprintf('X||%s', CSData.unit_cell_alignment.x);
        end

        if isfield(CSData.unit_cell_alignment, 'y')
            alignment{end + 1} = sprintf('Y||%s', CSData.unit_cell_alignment.y);
        end

        if isfield(CSData.unit_cell_alignment, 'z')
            alignment{end + 1} = sprintf('Z||%s', CSData.unit_cell_alignment.z);
        end

        CS{CSIdx} = crystalSymmetry(CSData.symmetry, 'mineral', CSData.mineral, alignment{:});
    end

    ebsd = EBSD.load( ...
        allArgs.orientations_data_path, ...
        'CS', CS, ...
        'SS', SS, ...
        'ColumnNames', {'Phase', 'x', 'y', 'Euler1', 'Euler2', 'Euler3'}, ...
        'Bunge' ...
    );

    grains = calcGrains(ebsd, 'FMC', allArgs.C_Maha);
    grains_smooth = smooth(grains, allArgs.smoothing);

    ebsdsq = gridify(ebsd);
    grainIDs = zeros(size(ebsdsq), 'int32');

    for xIdx = 1:size(ebsdsq.x, 1)

        for yIdx = 1:size(ebsdsq.x, 2)
            xCoord = ebsdsq.x(xIdx, yIdx);
            yCoord = ebsdsq.y(xIdx, yIdx);
            point = [xCoord, yCoord];
            grainID = findByLocation(grains_smooth, point);

            if numel(grainID) == 1
                grainIDs(xIdx, yIdx) = grainID;
            end

        end

    end

    save('grainIDs.mat', 'grainIDs');

    figure();
    h = heatmap(grainIDs);
    h.GridVisible = "off";
    colormap default
    exportgraphics(gcf, 'grainIDs.png');

    colours = cell(length(phaseNames));

    for phaseIdx = 1:length(phaseNames)
        phaseName = phaseNames{phaseIdx};
        colourMap = ones(grains_smooth(phaseName).length, 3) .* grains_smooth(phaseName).id / grains_smooth.length;
        colourMap(1:end, 1:2) = 0; % assign blue channel only
        colours{phaseIdx} = colourMap;
    end

    % These preferences then coincide with viewing the grain assignment in a
    % Python figure:
    setMTEXpref('xAxisDirection', 'east');
    setMTEXpref('zAxisDirection', 'intoPlane');

    % Show grain boundaries of clustered grains, overlayed on IPF map
    figure();

    for phaseIdx = 1:length(phaseNames)
        phaseName = phaseNames{phaseIdx};
        plot(ebsd(phaseName), ebsd(phaseName).orientations, 'micronbar', 'off');
        hold on;
    end

    plot(grains.boundary);
    hold off;
    set(gcf, 'graphicssmoothing', 'off')
    exportgraphics(gca, 'clustered_grains_original.png', 'Resolution', allArgs.fig_resolution);

    % Show *smoothed* grain boundaries of clustered grains, overlayed on IPF map
    figure();

    for phaseIdx = 1:length(phaseNames)
        phaseName = phaseNames{phaseIdx};
        plot(ebsd(phaseName), ebsd(phaseName).orientations, 'micronbar', 'off');
        hold on;
    end

    plot(grains_smooth.boundary);
    hold off;
    set(gcf, 'graphicssmoothing', 'off')
    exportgraphics(gca, 'clustered_grains_smoothed.png', 'Resolution', allArgs.fig_resolution);

    % Show mean orientation of clustered grains and grain boundaries
    figure();
    ipfKeys = cell(length(phaseNames));

    for phaseIdx = 1:length(phaseNames)
        phaseName = phaseNames{phaseIdx};
        ipfKey = ipfColorKey(ebsd(phaseName));
        ipfKeys{phaseIdx} = ipfKey;
        mean_ori_colors = ipfKey.orientation2color(grains(phaseName).meanOrientation);
        plot(grains_smooth(phaseName), mean_ori_colors, 'micronbar');
        hold on;
    end

    hold off;
    set(gcf, 'graphicssmoothing', 'off')
    exportgraphics(gca, 'clustered_grains_mean_ori.png', 'Resolution', allArgs.fig_resolution);

    % Save IPF keys
    for phaseIdx = 1:length(ipfKeys)
        phaseName = phaseNames{phaseIdx};
        figure();
        plot(ipfKeys{phaseIdx});
        exportgraphics(gca, sprintf('IPF_key_%s.png', phaseName), 'Resolution', allArgs.fig_resolution);
    end

    % Show grain boundary misorientation distribution
    figure();

    for phaseIdx_1 = 1:length(phaseNames)

        for phaseIdx_2 = phaseIdx_1:length(phaseNames)
            phaseName_1 = phaseNames{phaseIdx_1};
            phaseName_2 = phaseNames{phaseIdx_2};
            GB = grains.boundary(phaseName_1, phaseName_2);

            if ~isempty(GB)
                plotAngleDistribution(GB.misorientation);
                exportgraphics(gca, sprintf('misori_dist_%s_%s.png', phaseName_1, phaseName_2), 'Resolution', allArgs.fig_resolution);
            end

        end

    end

    % Show (smoothed) grain boundary misorientation distribution
    figure();

    for phaseIdx_1 = 1:length(phaseNames)

        for phaseIdx_2 = phaseIdx_1:length(phaseNames)
            phaseName_1 = phaseNames{phaseIdx_1};
            phaseName_2 = phaseNames{phaseIdx_2};
            GB = grains_smooth.boundary(phaseName_1, phaseName_2);

            if ~isempty(GB)
                plotAngleDistribution(GB.misorientation);
                exportgraphics(gca, sprintf('misori_dist_smooth_%s_%s.png', phaseName_1, phaseName_2), 'Resolution', allArgs.fig_resolution);
            end

        end

    end

end
