function run_omst_single_matrix(input_file, output_file)
    % run_omst_single_matrix.m
    %
    % Process a single 3D centroid coherence matrix with OMST filtering

    fprintf('Loading %s...\n', input_file);
    data = load(input_file);

    % ENSURE DOUBLE PRECISION
    cent_coh_mat = double(data.centroid_coherence_array);

    original_size = size(cent_coh_mat);
    fprintf('Original centroid coherence array size: [%d %d %d]\n', original_size(1), original_size(2), original_size(3));
    fprintf('Data type: %s\n', class(cent_coh_mat));

    if original_size(1) < original_size(2) && original_size(2) == original_size(3)
        cent_coh_mat = permute(cent_coh_mat, [2, 3, 1]);
        fprintf('Permuted from (K,N,N) to (N,N,K)\n');
    end

    s_cent = size(cent_coh_mat);
    N = s_cent(1);
    K = s_cent(3);

    fprintf('Matrix dimensions: N=%d ROIs, K=%d clusters\n', N, K);

    % Preallocate
    omst_pos_centroid_A = zeros(N, N, K);
    omst_neg_centroid_A = zeros(N, N, K);
    omst_centroid_A_ov_pos = cell(K,1);
    omst_centroid_A_ov_neg = cell(K,1);
    omst_centroid_B = zeros(N, N, K);
    omst_centroid_B_ov = cell(K,1);

    epsilon = 1e-12;

    % Strategy A
    fprintf('\nProcessing Strategy A...\n');
    for i = 1:K
        fprintf('  Cluster %d/%d\n', i, K);
        mtx_original = double(cent_coh_mat(:,:,i));  % Ensure double

        % Positive
        pos_mtx = mtx_original;
        pos_mtx(pos_mtx < 0) = 0;
        pos_mtx(pos_mtx == 0) = epsilon;
        pos_mtx = double(pos_mtx);  % Ensure double before OMST

        [nCIJtree_pos, CIJtree_pos, mdeg_pos, gce_pos, costmax_pos, E_pos] = ...
            threshold_omst_gce_wu_very_fast(pos_mtx, 0);

        omst_pos_centroid_A(:,:,i) = CIJtree_pos;
        omst_centroid_A_ov_pos{i} = struct('nCIJtree', nCIJtree_pos, ...
                                          'mdeg', mdeg_pos, ...
                                          'gce', gce_pos, ...
                                          'costmax', costmax_pos, ...
                                          'E', E_pos);

        % Negative
        neg_mtx = mtx_original;
        neg_mtx(neg_mtx > 0) = 0;
        neg_mtx = abs(neg_mtx);
        neg_mtx(neg_mtx == 0) = epsilon;
        neg_mtx = double(neg_mtx);  % Ensure double before OMST

        [nCIJtree_neg, CIJtree_neg, mdeg_neg, gce_neg, costmax_neg, E_neg] = ...
            threshold_omst_gce_wu_very_fast(neg_mtx, 0);

        omst_neg_centroid_A(:,:,i) = -CIJtree_neg;
        omst_centroid_A_ov_neg{i} = struct('nCIJtree', nCIJtree_neg, ...
                                          'mdeg', mdeg_neg, ...
                                          'gce', gce_neg, ...
                                          'costmax', costmax_neg, ...
                                          'E', E_neg);
    end

    % Strategy B
    fprintf('\nProcessing Strategy B...\n');
    for i = 1:K
        fprintf('  Cluster %d/%d\n', i, K);
        mtx_original = double(cent_coh_mat(:,:,i));  % Ensure double

        abs_mtx = abs(mtx_original);
        abs_mtx(abs_mtx == 0) = epsilon;
        abs_mtx = double(abs_mtx);  % Ensure double before OMST

        [nCIJtree_abs, CIJtree_abs, mdeg_abs, gce_abs, costmax_abs, E_abs] = ...
            threshold_omst_gce_wu_very_fast(abs_mtx, 0);

        sign_mtx = sign(mtx_original);
        CIJtree_signed = CIJtree_abs .* sign_mtx;

        omst_centroid_B(:,:,i) = CIJtree_signed;
        omst_centroid_B_ov{i} = struct('nCIJtree', nCIJtree_abs, ...
                                      'mdeg', mdeg_abs, ...
                                      'gce', gce_abs, ...
                                      'costmax', costmax_abs, ...
                                      'E', E_abs);
    end

    % Transpose back to (K, N, N) for Python
    omst_pos_centroid_A = permute(omst_pos_centroid_A, [3, 1, 2]);
    omst_neg_centroid_A = permute(omst_neg_centroid_A, [3, 1, 2]);
    omst_centroid_B = permute(omst_centroid_B, [3, 1, 2]);

    fprintf('\nSaving results to %s...\n', output_file);
    save(output_file, ...
        'omst_pos_centroid_A', 'omst_neg_centroid_A', ...
        'omst_centroid_A_ov_pos', 'omst_centroid_A_ov_neg', ...
        'omst_centroid_B', 'omst_centroid_B_ov', ...
        'N', 'K', ...
        '-v7');

    fprintf('Results saved successfully.\n');
    fprintf('Final matrix dimensions: %d x %d x %d (K x N x N)\n', K, N, N);
end