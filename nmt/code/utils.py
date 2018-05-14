
def buildBatchList(dataSize, batchSize):
    batchList = []
    if dataSize % batchSize == 0:
        numBatch = dataSize // batchSize
    else:
        numBatch = int(dataSize / batchSize) + 1

    for i in range(numBatch):
        batch = []
        batch.append(i * batchSize)
        if i == numBatch - 1:
            batch.append(dataSize - 1)
        else:
            batch.append((i + 1) * batchSize - 1)
        batchList.append(batch)

    return batchList


def write_attention_weights_to_file(attention_matrix, attention_file_path):
    context_attention = open(attention_file_path, "w")
    for attention_score in attention_matrix:
        for i in range(len(attention_score)):
            attention_weight = [str(float(weight))
                                for weight in attention_score[i]]
            context_attention.write(
                " ".join(attention_weight) + "\n")
        context_attention.write("\n")
