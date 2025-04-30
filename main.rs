use std::fs::File;
use std::io::{self, BufRead, BufReader};
use plotters::prelude::*;

// Define a simple data structure to hold our data
#[derive(Debug, Clone)]
struct DataPoint {
    features: Vec<f64>,
    label: bool,
}

// Struktur untuk menyimpan parameter normalisasi
#[derive(Debug)]
struct Normalizer {
    min_values: Vec<f64>,
    max_values: Vec<f64>,
}

// Function to read CSV data
fn read_csv(path: &str) -> io::Result<Vec<DataPoint>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    
    // Skip the header
    let mut lines = reader.lines();
    let _ = lines.next();
    
    for line in lines {
        let line = line?;
        let values: Vec<&str> = line.split(',').collect();
        
        if values.len() >= 5 {
            let features = vec![
                values[0].parse::<f64>().unwrap_or(0.0),  // Temperature
                values[1].parse::<f64>().unwrap_or(0.0),  // Humidity
                values[2].parse::<f64>().unwrap_or(0.0),  // Soil_pH
                values[3].parse::<f64>().unwrap_or(0.0),  // Soil_Moisture
            ];
            
            let label = values[4].parse::<u8>().unwrap_or(0) == 1;
            
            data.push(DataPoint {
                features,
                label,
            });
        }
    }
    
    Ok(data)
}

// Simple normalization function to scale features
fn normalize_data(data: &mut Vec<DataPoint>) -> Normalizer {
    // Find min and max for each feature
    let mut min_values = vec![f64::MAX; 4];
    let mut max_values = vec![f64::MIN; 4];
    
    for point in data.iter() {
        for (i, &value) in point.features.iter().enumerate() {
            min_values[i] = min_values[i].min(value);
            max_values[i] = max_values[i].max(value);
        }
    }
    
    // Apply min-max normalization
    for point in data.iter_mut() {
        for i in 0..point.features.len() {
            let range = max_values[i] - min_values[i];
            if range > 0.0 {
                point.features[i] = (point.features[i] - min_values[i]) / range;
            }
        }
    }
    
    Normalizer {
        min_values,
        max_values,
    }
}

// Normalize a single feature vector using the normalizer
fn normalize_features(features: &[f64], normalizer: &Normalizer) -> Vec<f64> {
    let mut normalized = features.to_vec();
    for i in 0..features.len() {
        let range = normalizer.max_values[i] - normalizer.min_values[i];
        if range > 0.0 {
            normalized[i] = (features[i] - normalizer.min_values[i]) / range;
        }
    }
    normalized
}

// Split data into training and testing sets
fn train_test_split(data: &[DataPoint], test_ratio: f64) -> (Vec<DataPoint>, Vec<DataPoint>) {
    let test_size = (data.len() as f64 * test_ratio) as usize;
    let mut training = Vec::with_capacity(data.len() - test_size);
    let mut testing = Vec::with_capacity(test_size);
    
    // Simple split (not random for simplicity)
    for (i, point) in data.iter().enumerate() {
        if i % 5 == 0 {  // Every 5th element goes to test set (~20%)
            testing.push(point.clone());
        } else {
            training.push(point.clone());
        }
    }
    
    (training, testing)
}

// Calculate euclidean distance between two points
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// KNN Implementation
struct KNN {
    k: usize,
    training_data: Vec<DataPoint>,
}

impl KNN {
    fn new(k: usize) -> Self {
        KNN {
            k,
            training_data: Vec::new(),
        }
    }
    
    fn fit(&mut self, training_data: Vec<DataPoint>) {
        self.training_data = training_data;
    }
    
    fn predict(&self, features: &[f64]) -> bool {
        // Calculate distances to all training points
        let mut distances: Vec<(f64, bool)> = self.training_data
            .iter()
            .map(|point| (euclidean_distance(&point.features, features), point.label))
            .collect();
        
        // Sort by distance
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Take the k nearest neighbors
        let k_nearest = &distances[0..self.k.min(distances.len())];
        
        // Count votes
        let mut true_count = 0;
        let mut false_count = 0;
        
        for &(_, label) in k_nearest {
            if label {
                true_count += 1;
            } else {
                false_count += 1;
            }
        }
        
        // Return the majority vote
        true_count > false_count
    }
    
    fn evaluate(&self, test_data: &[DataPoint]) -> f64 {
        let mut correct = 0;
        
        for point in test_data {
            let prediction = self.predict(&point.features);
            if prediction == point.label {
                correct += 1;
            }
        }
        
        correct as f64 / test_data.len() as f64
    }
}

// Simple SVM Implementation (Linear SVM)
struct SVM {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    iterations: usize,
}

impl SVM {
    fn new(feature_count: usize, learning_rate: f64, iterations: usize) -> Self {
        SVM {
            weights: vec![0.0; feature_count],
            bias: 0.0,
            learning_rate,
            iterations,
        }
    }
    
    fn fit(&mut self, training_data: &[DataPoint]) {
        for _ in 0..self.iterations {
            for point in training_data {
                // Calculate prediction
                let label = if point.label { 1.0 } else { -1.0 };
                let prediction = self.predict_raw(&point.features);
                
                // Update weights if prediction is wrong
                if label * prediction <= 0.0 {
                    // Update weights and bias
                    for i in 0..self.weights.len() {
                        self.weights[i] += self.learning_rate * label * point.features[i];
                    }
                    self.bias += self.learning_rate * label;
                }
            }
        }
    }
    
    fn predict_raw(&self, features: &[f64]) -> f64 {
        let mut result = self.bias;
        for (i, &feature) in features.iter().enumerate() {
            result += self.weights[i] * feature;
        }
        result
    }
    
    fn predict(&self, features: &[f64]) -> bool {
        self.predict_raw(features) > 0.0
    }
    
    fn evaluate(&self, test_data: &[DataPoint]) -> f64 {
        let mut correct = 0;
        
        for point in test_data {
            let prediction = self.predict(&point.features);
            if prediction == point.label {
                correct += 1;
            }
        }
        
        correct as f64 / test_data.len() as f64
    }
}

// Function to visualize the dataset with 2 features
fn plot_data_visualization(data: &[DataPoint], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create a drawing area
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Find the min and max values for pH and moisture
    let mut min_ph = f64::MAX;
    let mut max_ph = f64::MIN;
    let mut min_moisture = f64::MAX;
    let mut max_moisture = f64::MIN;
    
    for point in data {
        // Using pH (index 2) and Moisture (index 3) for visualization
        let ph = point.features[2];
        let moisture = point.features[3];
        
        min_ph = min_ph.min(ph);
        max_ph = max_ph.max(ph);
        min_moisture = min_moisture.min(moisture);
        max_moisture = max_moisture.max(moisture);
    }
    
    // Add some margin
    let margin = 0.1;
    min_ph -= margin;
    max_ph += margin;
    min_moisture -= margin;
    max_moisture += margin;
    
    // Create the chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Soil Fertility Classification", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_ph..max_ph, min_moisture..max_moisture)?;
    
    // Configure the chart
    chart
        .configure_mesh()
        .x_desc("Soil pH")
        .y_desc("Soil Moisture")
        .draw()?;
    
    // Plot the data points based on labels
    chart.draw_series(
        data.iter().filter(|p| p.label).map(|p| {
            Circle::new((p.features[2], p.features[3]), 3, GREEN.filled())
        }),
    )?
    .label("Fertile")
    .legend(|(x, y)| Circle::new((x, y), 3, GREEN.filled()));
    
    chart.draw_series(
        data.iter().filter(|p| !p.label).map(|p| {
            Circle::new((p.features[2], p.features[3]), 3, RED.filled())
        }),
    )?
    .label("Not Fertile")
    .legend(|(x, y)| Circle::new((x, y), 3, RED.filled()));
    
    chart.configure_series_labels()
        .background_style(WHITE.filled())
        .border_style(&BLACK)
        .draw()?;
    
    Ok(())
}

// Function to visualize the decision boundaries for KNN and SVM
fn plot_decision_boundaries(
    svm: &SVM,
    knn: &KNN,
    data: &[DataPoint],
    normalizer: &Normalizer,
    filename_knn: &str,
    filename_svm: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // === Setup untuk KNN ===
    let root_knn = BitMapBackend::new(filename_knn, (800, 600)).into_drawing_area();
    root_knn.fill(&WHITE)?;

    // === Setup untuk SVM ===
    let root_svm = BitMapBackend::new(filename_svm, (800, 600)).into_drawing_area();
    root_svm.fill(&WHITE)?;

    // === Tentukan rentang pH dan kelembapan ===
    let mut min_ph = f64::MAX;
    let mut max_ph = f64::MIN;
    let mut min_moisture = f64::MAX;
    let mut max_moisture = f64::MIN;

    for point in data {
        let ph = point.features[2];
        let moisture = point.features[3];
        min_ph = min_ph.min(ph);
        max_ph = max_ph.max(ph);
        min_moisture = min_moisture.min(moisture);
        max_moisture = max_moisture.max(moisture);
    }

    // Tambahkan margin
    let margin = 0.1;
    min_ph -= margin;
    max_ph += margin;
    min_moisture -= margin;
    max_moisture += margin;

    // === Buat chart untuk KNN ===
    let mut chart_knn = ChartBuilder::on(&root_knn)
        .caption("KNN Decision Boundaries", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_ph..max_ph, min_moisture..max_moisture)?;

    chart_knn
        .configure_mesh()
        .x_desc("Soil pH")
        .y_desc("Soil Moisture")
        .draw()?;

    // === Buat chart untuk SVM ===
    let mut chart_svm = ChartBuilder::on(&root_svm)
        .caption("SVM Decision Boundaries", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_ph..max_ph, min_moisture..max_moisture)?;

    chart_svm
        .configure_mesh()
        .x_desc("Soil pH")
        .y_desc("Soil Moisture")
        .draw()?;

    // === Buat grid untuk prediksi ===
    const GRID_SIZE: usize = 40;
    let step_x = (max_ph - min_ph) / GRID_SIZE as f64;
    let step_y = (max_moisture - min_moisture) / GRID_SIZE as f64;

    // Simpan prediksi KNN dan SVM
    let mut knn_mesh = vec![];
    let mut svm_mesh = vec![];

    for i in 0..GRID_SIZE {
        for j in 0..GRID_SIZE {
            let x = min_ph + i as f64 * step_x;
            let y = min_moisture + j as f64 * step_y;

            // Gunakan nilai rata-rata untuk suhu dan kelembapan (sebelum normalisasi)
            let feature_point = vec![29.0, 63.0, x, y];
            // Normalisasi fitur menggunakan normalizer
            let normalized_features = normalize_features(&feature_point, normalizer);

            // Prediksi untuk KNN
            let knn_prediction = knn.predict(&normalized_features);
            knn_mesh.push((x, y, knn_prediction));

            // Prediksi untuk SVM
            let svm_prediction = svm.predict(&normalized_features);
            svm_mesh.push((x, y, svm_prediction));
        }
    }

    // === Gambar wilayah keputusan KNN ===
    chart_knn.draw_series(
        knn_mesh.iter().map(|&(x, y, prediction)| {
            let color = if prediction {
                GREEN.mix(0.2)
            } else {
                RED.mix(0.2)
            };
            Rectangle::new(
                [(x - step_x / 2.0, y - step_y / 2.0), (x + step_x / 2.0, y + step_y / 2.0)],
                color.filled(),
            )
        }),
    )?;

    // === Gambar wilayah keputusan SVM ===
    chart_svm.draw_series(
        svm_mesh.iter().map(|&(x, y, prediction)| {
            let color = if prediction {
                GREEN.mix(0.2)
            } else {
                RED.mix(0.2)
            };
            Rectangle::new(
                [(x - step_x / 2.0, y - step_y / 2.0), (x + step_x / 2.0, y + step_y / 2.0)],
                color.filled(),
            )
        }),
    )?;

    // === Gambar titik data untuk kedua plot ===
    for chart in [&mut chart_knn, &mut chart_svm].iter_mut() {
        chart.draw_series(
            data.iter().map(|p| {
                let color = if p.label { GREEN } else { RED };
                Circle::new((p.features[2], p.features[3]), 3, color.filled())
            }),
        )?;
    }

    // === Tambahkan label seri ===
    for chart in [&mut chart_knn, &mut chart_svm].iter_mut() {
        chart
            .configure_series_labels()
            .background_style(WHITE.filled())
            .border_style(&BLACK)
            .draw()?;
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to the CSV file
    let path = "data/chili_soil_data.csv";
    
    // Load data
    let mut data = read_csv(path)?;
    
    // Normalize data and get normalizer
    let normalizer = normalize_data(&mut data);
    
    // Split data
    let (training_data, test_data) = train_test_split(&data, 0.2);
    
    println!("Training set size: {}", training_data.len());
    println!("Test set size: {}", test_data.len());
    
    // Train and evaluate KNN
    let mut knn = KNN::new(5);  // k=5 neighbors
    knn.fit(training_data.clone());
    let knn_accuracy = knn.evaluate(&test_data);
    println!("KNN Accuracy: {:.2}%", knn_accuracy * 100.0);
    
    // Train and evaluate SVM
    let mut svm = SVM::new(4, 0.01, 1000);  // 4 features, learning rate 0.01, 1000 iterations
    svm.fit(&training_data);
    let svm_accuracy = svm.evaluate(&test_data);
    println!("SVM Accuracy: {:.2}%", svm_accuracy * 100.0);
    
    // Test with a new data point
    let new_data = vec![29.0, 63.0, 6.1, 61.0];  // Example new data point
    let normalized_new_data = normalize_features(&new_data, &normalizer);
    let knn_prediction = knn.predict(&normalized_new_data);
    let svm_prediction = svm.predict(&normalized_new_data);
    
    println!("KNN prediction for new data: {}", if knn_prediction { "Fertile" } else { "Not Fertile" });
    println!("SVM prediction for new data: {}", if svm_prediction { "Fertile" } else { "Not Fertile" });
    
    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;
    
    // Create visualizations
    plot_data_visualization(&data, "output/data_visualization.png")?;
    plot_decision_boundaries(
        &svm,
        &knn,
        &data,
        &normalizer,
        "output/knn_decision_boundaries.png",
        "output/svm_decision_boundaries.png",
    )?;
    
    println!("Visualizations saved to output/data_visualization.png, output/knn_decision_boundaries.png, and output/svm_decision_boundaries.png");
    
    Ok(())
}