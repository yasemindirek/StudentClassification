﻿using System;
using System.IO;
using Microsoft.ML;
using StudentClassification.MachineLearning.DataModels;


namespace StudentClassification.MachineLearning.Predictors
{
    /// <summary>
    /// This class is used to load the saved model and run some predictions.
    /// The model is loaded from a defined file, and predictions are made on the new sample.
    /// </summary>
    public class Predictor
    {
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "classification.mdl");
        private readonly MLContext _mlContext;

        private ITransformer _model;

        public Predictor()
        {
            _mlContext = new MLContext(111);
        }

        /// <summary>
        /// Runs prediction on new data.
        /// </summary>
        /// <param name="newSample">New data sample.</param>
        /// <returns>StudentsData object, which contains predictions made by model.</returns>
        public StudentsPrediction Predict(StudentsData newSample)
        {
            LoadModel();

            var predictionEngine = _mlContext.Model.CreatePredictionEngine<StudentsData, StudentsPrediction>(_model);

            return predictionEngine.Predict(newSample);
        }

        private void LoadModel()
        {
            if (!File.Exists(ModelPath))
            {
                throw new FileNotFoundException($"File {ModelPath} doesn't exist.");
            }

            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                _model = _mlContext.Model.Load(stream, out _);
            }

            if (_model == null)
            {
                throw new Exception($"Failed to load Model");
            }
        }
    }
}
