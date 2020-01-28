from pyspark.ml.tuning import CrossValidator

def tuning(model):
  '''
  CrossValidator from pyspark.ml.tuning doesn't give the best Hyper-parameter setting.
  This is a function to output the performance of different Hyper-parameter sets
  
  input: a CrossValidator fitted model
  output: a dictionary with keys: hyper-para settings, values: metrics
          sorted by value
  '''
  length = len(model.avgMetrics)
  res = {}
  for i in range(length):
    s = ""
    paraDict = model.extractParamMap()[model.estimatorParamMaps][i]
    for j in paraDict.keys():
      s += str(j).split("__")[1] + "  " + str(paraDict[j]) + "  "
    res[s.strip()] = model.avgMetrics[i]
  return {k: v for k, v in sorted(res.items(), key=lambda item: item[1])}
