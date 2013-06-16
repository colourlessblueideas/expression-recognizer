% The main function
% If test_images is provided, it will predict the results for those too, otherwise predicts 0 for the test cases.

load training.mat;
load val_images;
% load test_images;

h = size(tr_images,1);
w = size(tr_images,2);

if ~exist('test_images', 'var')
  test_images = val_images;
else
  test_images = cat(3, val_images, test_images);
end


% Cross validation
for K=[3:10 15 20 35 50]
  nfold = 10;
  acc(K) = cross_validate(K, tr_images(:,:,1:2507), tr_labels(1:2507), nfold);
  fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K, acc(K));
end
[maxacc bestK] = max(acc);
fprintf('K is selected to be %d.\n', bestK);
% I get a bestK of 5

% Run the classifier
prediction = knn_classifier(bestK, tr_images(:,:,1:2507), tr_labels, tr_images(:,:,2508:end));


% Fill in the test labels with 0 if necessary
if (length(prediction) < 1253)
  prediction = [prediction; zeros(1253-length(prediction), 1)];
end

numCorrect=0;
for k=2508:2925
if prediction(k-2507) == tr_labels(k)
    numCorrect = numCorrect + 1;
end
end

fprintf('overall accuracy: %d/418\n',numCorrect);

% Print the predictions to file
fprintf('writing the output to prediction.csv\n');
fid = fopen('prediction.csv', 'w');
for i=1:length(prediction)
  fprintf(fid, '%d\n', prediction(i));
end
fclose(fid);

clear tr_images test_images val_images
