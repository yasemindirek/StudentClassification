using System;
using Microsoft.ML.Data;

namespace StudentClassification.MachineLearning.DataModels
{
    /// <summary>
    /// Dataset Description
    /// </summary>
    public class StudentsData
    {
        [LoadColumn(0)]
        public float CorrectAnswer { get; set; } // First column: Number of correct answers

        [LoadColumn(1)]
        public float AnswerTime { get; set; } // Second column: Total Answer Time of the student

        [LoadColumn(2)]
        public string Label { get; set; }  // Success of the student

    }
}
