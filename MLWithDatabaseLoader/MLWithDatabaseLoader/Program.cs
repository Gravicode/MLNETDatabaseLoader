using Microsoft.ML;
using Microsoft.ML.Data;
using MLWithDatabaseLoader.Model;
using System;
using System.Data.SqlClient;
using Microsoft.ML.Transforms;
using System.Linq;
using System.IO;
using System.Diagnostics;
using Helper;
using System.Drawing;
using System.Data;
using System.Collections.Generic;

namespace MLWithDatabaseLoader
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // localdb SQL database connection string using a filepath to attach the database file into localdb
            string dbFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Database", "Iris.mdf");
            string connectionString = $"Data Source = (LocalDB)\\MSSQLLocalDB;AttachDbFilename={dbFilePath};Database=Iris;Integrated Security = True";

            // ConnString Example: localdb SQL database connection string for 'localdb default location' (usually files located at /Users/YourUser/)
            //string connectionString = @"Data Source=(localdb)\MSSQLLocalDb;Initial Catalog=YOUR_DATABASE;Integrated Security=True;Pooling=False";
            //
            // ConnString Example: on-premises SQL Server Database (Integrated security)
            //string connectionString = @"Data Source=YOUR_SERVER;Initial Catalog=YOUR_DATABASE;Integrated Security=True;Pooling=False";
            //
            // ConnString Example:  Azure SQL Database connection string
            //string connectionString = @"Server=tcp:yourserver.database.windows.net,1433; Initial Catalog = YOUR_DATABASE; Persist Security Info = False; User ID = YOUR_USER; Password = YOUR_PASSWORD; MultipleActiveResultSets = False; Encrypt = True; TrustServerCertificate = False; Connection Timeout = 60; ConnectRetryCount = 5; ConnectRetryInterval = 10;";

            string commandText = "SELECT * from IrisData";

            DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<IrisData>();

            DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance,
                                                         connectionString,
                                                         commandText);

            IDataView dataView = loader.Load(dbSource);
            var pre = dataView.Preview();

            var trainTestData = mlContext.Data.TrainTestSplit(dataView);
            var finalTransformerPipeLine = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "class", outputColumnName: "KeyColumn").
                Append(mlContext.Transforms.Concatenate("Features", nameof(IrisData.petal_length), nameof(IrisData.petal_width), nameof(IrisData.sepal_length),
                nameof(IrisData.sepal_width)));



            // Apply the ML algorithm
            var trainingPipeLine = finalTransformerPipeLine.Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "KeyColumn", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "class", inputColumnName: "KeyColumn"));

            Console.WriteLine("Training the ML model while streaming data from a SQL database...");
            Stopwatch watch = new Stopwatch();
            watch.Start();

            var model = trainingPipeLine.Fit(trainTestData.TrainSet);

            watch.Stop();
            Console.WriteLine("Elapsed time for training the model = {0} seconds", watch.ElapsedMilliseconds / 1000);

            Console.WriteLine("Evaluating the model...");
            Stopwatch watch2 = new Stopwatch();
            watch2.Start();

            var predictions = model.Transform(trainTestData.TestSet);
            // Now that we have the test predictions, calculate the metrics of those predictions and output the results.
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "KeyColumn", "Score");

            watch2.Stop();
            Console.WriteLine("Elapsed time for evaluating the model = {0} seconds", watch2.ElapsedMilliseconds / 1000);

            ConsoleHelper.PrintMultiClassClassificationMetrics("==== Evaluation Metrics training from a Database ====", metrics);
            
            Console.WriteLine("Trying a single prediction:");

            var predictionEngine = mlContext.Model.CreatePredictionEngine<IrisData, DataPrediction>(model);

            var sampleData1 = new IrisData()
            {
                sepal_length = 6.1f,
                sepal_width = 3f,
                petal_length = 4.9f,
                petal_width = 1.8f,
                class1 = string.Empty
            };

            var sampleData2 = new IrisData()
            {
                sepal_length = 5.1f,
                sepal_width = 3.5f,
                petal_length = 1.4f,
                petal_width = 0.2f,
                class1 = string.Empty
            };

            var irisPred1 = predictionEngine.Predict(sampleData1);
            var irisPred2 = predictionEngine.Predict(sampleData2);
        
            // Since we apply MapValueToKey estimator with default parameters, key values
            // depends on order of occurence in data file. Which is "Iris-setosa", "Iris-versicolor", "Iris-virginica"
            // So if we have Score column equal to [0.2, 0.3, 0.5] that's mean what score for
            // Iris-setosa is 0.2
            // Iris-versicolor is 0.3
            // Iris-virginica is 0.5.
            //Add a dictionary to map the above float values to strings. 
            Dictionary<float, string> IrisFlowers = new Dictionary<float, string>();
            IrisFlowers.Add(0, "Setosa");
            IrisFlowers.Add(1, "versicolor");
            IrisFlowers.Add(2, "virginica");

            Console.WriteLine($"Predicted Label 1: {IrisFlowers[Array.IndexOf(irisPred1.Score, irisPred1.Score.Max())]} - Score:{irisPred1.Score.Max()}", Color.YellowGreen);
            Console.WriteLine($"Predicted Label 2: {IrisFlowers[Array.IndexOf(irisPred2.Score, irisPred2.Score.Max())]} - Score:{irisPred2.Score.Max()}", Color.YellowGreen);
            Console.WriteLine();

            //*** Detach database from localdb only if you used a conn-string with a filepath to attach the database file into localdb ***
            Console.WriteLine("... Detaching database from SQL localdb ...");
            DetachDatabase(connectionString);

            Console.WriteLine("=============== Press any key ===============");
            Console.ReadKey();

        }
        public static void DetachDatabase(string userConnectionString) //DELETE PARAM *************
        {
            string dbName = string.Empty;
            using (SqlConnection userSqlDatabaseConnection = new SqlConnection(userConnectionString))
            {
                userSqlDatabaseConnection.Open();
                dbName = userSqlDatabaseConnection.Database;
            }

            string masterConnString = $"Data Source = (LocalDB)\\MSSQLLocalDB;Integrated Security = True";
            using (SqlConnection sqlDatabaseConnection = new SqlConnection(masterConnString))
            {
                sqlDatabaseConnection.Open();

                string prepareDbcommandString = $"ALTER DATABASE [{dbName}] SET OFFLINE WITH ROLLBACK IMMEDIATE ALTER DATABASE [{dbName}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE";
                //(ALTERNATIVE) string prepareDbcommandString = $"ALTER DATABASE [{dbName}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE";
                SqlCommand sqlPrepareCommand = new SqlCommand(prepareDbcommandString, sqlDatabaseConnection);
                sqlPrepareCommand.ExecuteNonQuery();

                string detachCommandString = "sp_detach_db";
                SqlCommand sqlDetachCommand = new SqlCommand(detachCommandString, sqlDatabaseConnection);
                sqlDetachCommand.CommandType = CommandType.StoredProcedure;
                sqlDetachCommand.Parameters.AddWithValue("@dbname", dbName);
                sqlDetachCommand.ExecuteNonQuery();
            }
        }
    }
}
