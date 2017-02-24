import matplotlib as plt

# global variable
thr_array = np.linspace(0,1,100)

def computeAP(recall_array,precision_array):
    assert precision_array.shape == recall_array.shape, "in compute AP, shape mismatch"

    result = 0
    for k in range(len(precision_array)-1):
        result += precision_array[k] * (recall_array[k]-recall_array[k+1])
    assert result <= 1 and result >= 0, "in computeAP, result non consistant"

    return result

def plot(pr,output_folder,output_name):
    # pr : dict{title:[[recall values][precision values]]}
    # path to output the graph as png
    title = "PR curves"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fig = plt.figure(figsize=(12,10))
    ax = plt.subplot(111)
    plt.grid()
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.title(title)
    for key in pr.keys():
        mAP = computeAP(pr[key][0,:],pr[key][1,:])
        plt.plot(pr[key][0,:],pr[key][1,:],label="%s : %.2f" % (key,mAP))

    # take the mean
    pr_values = np.sum(np.array(pr.values()),axis=0)
    recalls = pr_values[0,:]
    precisions = pr_values[1,:]

    recall_mean = recalls / len(recalls)
    precision_mean = precisions / len(precisions)


    mAP = computeAP(recall_mean,precision_mean)

    plt.plot(recall_mean,precision_mean,label="%s : %.2f" % ("mean",mAP))

    # Shrink current axis by 20%
    plt.tight_layout()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':10})

    plt.savefig(os.path.join(output_folder,output_name))
    print "Figure saved at %s" % os.path.join(output_folder,output_name)

def get_PR_coordinates(predictions,gt):
    # predictions is a dict{method_name:[predictions_array]}
    pr = {}
    for method in predictions.keys():

        pos_array = np.zeros(len(thr_array))
        true_pos_array = np.zeros(len(thr_array))

        for thr_index in range(len(thr_array)):
            thr = thr_array[thr_index]

            filtered_result = np.array(predictions[method]>=thr).astype(int) # 0 or 1
            pos_array[thr_index] += np.sum(filtered_result)
            true_pos_array[thr_index] += np.sum(filtered_result*gt)

        precision = true_pos_array / pos_array
        recall = true_pos_array / np.sum(gt)

        pr[method] = np.vstack((recall,precision))

    return pr