using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLWithDatabaseLoader.Model
{
    public class IrisData
    {
        public float sepal_length { get; set; }
        public float sepal_width { get; set; }
        public float petal_length { get; set; }
        public float petal_width { get; set; }

        [ColumnName("class")]
        public string class1 { get; set; }
    }
}
