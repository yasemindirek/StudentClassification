using System;
using Microsoft.ML.Data;

namespace StudentClassification.MachineLearning.DataModels
{
    /// <summary>
    /// Students Prediction.
    /// </summary>
    public class StudentsPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }
}
