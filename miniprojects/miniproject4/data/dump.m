function dump ( type, batches, batch_size, data, labels)

count = 0;
for i = 0 : batches * batch_size : length(data) - 2*batch_size
    fprintf('Saving down %s batch %s\n', type, num2str(count));    
    %t =  i + batches * batch_size
    %size(data)
    x = data   ( i + 1 : ( i + batches * batch_size ), : );
    y = labels ( i + 1 : ( i + batches * batch_size ) );
    save(strcat(type,'/batch_',num2str(count),'.mat') ,'x', 'y','-v6');
    count = count + 1;    
end
end