function [ predictions ] = predictFace( model_sample, test_images )
%Returns predictions based on model.
%Assumes test_images is loaded.

ntest = size(test_images, 3);
h = size(test_images,1);
w = size(test_images,2);
test_images_face = double(reshape(test_images, [h*w, ntest]))';

testing_label_face = zeros(ntest,1);

%Make predictions for each trained model
for l=1:5
     [predict_label_sample(:,l), accuracy_sample(:,l), prob_estimates_sample(:,:,l)] = svmpredict(testing_label_face, test_images_face, model_sample(l), '-b 0');
end

numCorrectBagging = 0;

%Find the mode of the predictions of each model
for m=1:ntest
    predictions(m) = mode(predict_label_sample(m,:));
if mode(predict_label_sample(m,:)) == testing_label_face(m)
    numCorrectBagging = numCorrectBagging + 1;
end
end

fprintf('accuracy: %d/%d correct\n',numCorrectBagging,ntest);

save('test.mat', 'predictions');

end