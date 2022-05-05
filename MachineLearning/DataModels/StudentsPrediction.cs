using System;
using Microsoft.ML.Data;

namespace StudentClassification.MachineLearning.DataModels
{
    /// <summary>
    /// Models Palmer Penguins Binary Prediction.
    /// </summary>
    public class StudentsPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }
}
