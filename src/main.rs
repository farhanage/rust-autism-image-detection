use candle_core::{DType, Result, Tensor, ModuleT};
use candle_nn::{loss, ops, Conv2d, Linear, Optimizer, VarBuilder, VarMap};
use image;
use std::fs;
use std::path::{Path};
use ndarray::{Array2, Array1};
use ndarray_npy::{write_npy, read_npy};

mod config;
use config::Config;

const IMAGE_SIZE: usize = 224; // 64x64 face images
const LABELS: usize = 2; // autistic, non_autistic
const BSIZE: usize = 32; // Small batch size for low memory

// CNN for face detection
struct ConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: candle_nn::Dropout,
}

impl ConvNet {
    fn new(vs: VarBuilder) -> Result<Self> {
        // Consider using 3x3 convs with padding for better feature extraction
        let conv1 = candle_nn::conv2d(
            1, 
            16, 
            3, 
            candle_nn::Conv2dConfig { padding: 1, ..Default::default() }, 
            vs.pp("c1"))?;

        let conv2 = candle_nn::conv2d(
            16, 
            32, 
            3, 
            candle_nn::Conv2dConfig { padding: 1, ..Default::default() }, 
            vs.pp("c2"))?;

        // With padding=1: 224->224->112->112->56, so 32*56*56=100352
        let fc1 = candle_nn::linear(
            100352, 
            256, 
            vs.pp("fc1"))?;

        let fc2 = candle_nn::linear(
            256, 
            LABELS, 
            vs.pp("fc2"))?;

        let dropout = candle_nn::Dropout::new(0.3);

        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
        })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (b_sz, _img_dim) = xs.dims2()?;
        let xs = xs
            .reshape((b_sz, 1, IMAGE_SIZE, IMAGE_SIZE))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        self.dropout.forward_t(&xs, train)?.apply(&self.fc2)
    }
}

struct TrainingArgs {
    learning_rate: f64,
    epochs: usize,
    save: Option<String>,
    load: Option<String>,
}

// Preprocess images from raw to processed as .npy arrays, then load processed for training
fn preprocess_and_save(data_raw: &Path, data_processed: &Path) -> anyhow::Result<()> {
    let splits = ["train", "valid", "test"];
    let class_map = [("non_autistic", 0u32), ("autistic", 1u32)];
    for split in splits.iter() {
        let mut images = Vec::new();
        let mut labels = Vec::new();
        for (class, label) in class_map.iter() {
            let dir = data_raw.join(split).join(class);
            if !dir.exists() {
                println!("Warning: directory {:?} does not exist", dir);
                continue;
            }
            for entry in fs::read_dir(&dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map(|e| e == "jpg" || e == "png").unwrap_or(false) {
                    let img = image::open(&path)?.resize_exact(IMAGE_SIZE as u32, IMAGE_SIZE as u32, image::imageops::FilterType::Triangle).to_luma8();
                    let img: Vec<f32> = img.pixels().map(|p| p[0] as f32 / 255.0).collect();
                    images.push(img);
                    labels.push(*label);
                }
            }
        }
        let n = labels.len();
        if n == 0 {
            println!("No images found for split {split}");
            continue;
        }
        let images_arr = Array2::from_shape_vec((n, IMAGE_SIZE * IMAGE_SIZE), images.into_iter().flatten().collect())?;
        let labels_arr = Array1::from_vec(labels);
        let out_dir = data_processed.join(split);
        fs::create_dir_all(&out_dir)?;
        write_npy(out_dir.join("images.npy"), &images_arr)?;
        write_npy(out_dir.join("labels.npy"), &labels_arr)?;
        println!("Saved {n} images for split {split} to {:?}", out_dir);
    }
    Ok(())
}

fn load_processed(data_processed: &Path) -> anyhow::Result<((Tensor, Tensor), (Tensor, Tensor), (Tensor, Tensor))> {
    let splits = ["train", "valid", "test"];
    let mut tensors = Vec::new();
    for split in splits.iter() {
        let dir = data_processed.join(split);
        let images_path = dir.join("images.npy");
        let labels_path = dir.join("labels.npy");
        if !images_path.exists() || !labels_path.exists() {
            anyhow::bail!("Missing processed data for split {split} in {:?}", dir);
        }
        let images_arr: Array2<f32> = read_npy(images_path)?;
        let labels_arr: Array1<u32> = read_npy(labels_path)?;
        let n = labels_arr.len();
        let images = Tensor::from_vec(images_arr.into_raw_vec(), (n, IMAGE_SIZE * IMAGE_SIZE), &candle_core::Device::Cpu)?;
        let labels = Tensor::from_vec(labels_arr.to_vec(), n, &candle_core::Device::Cpu)?;
        tensors.push((images, labels));
    }
    Ok((tensors[0].clone(), tensors[1].clone(), tensors[2].clone()))
}

fn training_loop_cnn(
    train_images: Tensor,
    train_labels: Tensor,
    valid_images: Tensor,
    valid_labels: Tensor,
    test_images: Tensor,
    test_labels: Tensor,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    // Try to use CUDA, else fallback to CPU, and print error if CUDA not available
    let dev = match candle_core::Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => {
            println!("Using CUDA device for training.");
            d
        },
        Ok(d) => {
            eprintln!("Warning: CUDA not available, using CPU.");
            d
        },
        Err(e) => {
            eprintln!("Error initializing CUDA: {e}. Using CPU.");
            candle_core::Device::Cpu
        }
    };
    let train_images = train_images.to_dtype(DType::F32)?;
    let train_labels = train_labels.to_dtype(DType::U32)?;
    let valid_images = valid_images.to_dtype(DType::F32)?;
    let valid_labels = valid_labels.to_dtype(DType::U32)?;
    let test_images = test_images.to_dtype(DType::F32)?;
    let test_labels = test_labels.to_dtype(DType::U32)?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = ConvNet::new(vs.clone())?;

    if let Some(load) = &args.load {
        println!("Loading weights from {load}");
        varmap.load(load)?;
    }

    let adamw_params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;
    let n_batches = train_images.dim(0)? / BSIZE;
    for epoch in 1..=args.epochs {
        let mut sum_loss = 0f32;
        for batch_idx in 0..n_batches {
            let batch_images = train_images.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let batch_labels = train_labels.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let batch_images = batch_images.to_device(&dev)?;
            let batch_labels = batch_labels.to_device(&dev)?;
            let logits = model.forward(&batch_images, true)?;
            let log_sm = ops::log_softmax(&logits, candle_core::D::Minus1)?;
            let loss = loss::nll(&log_sm, &batch_labels)?;
            opt.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;
        }
        let avg_loss = sum_loss / n_batches as f32;

        // Test accuracy (batched)
        let mut test_ok = 0f32;
        let mut test_total = 0;
        let n_test = test_images.dim(0)?;
        let n_test_batches = (n_test + BSIZE - 1) / BSIZE;
        for batch_idx in 0..n_test_batches {
            let batch_size = if batch_idx == n_test_batches - 1 { n_test - batch_idx * BSIZE } else { BSIZE };
            let batch_images = test_images.narrow(0, batch_idx * BSIZE, batch_size)?;
            let batch_labels = test_labels.narrow(0, batch_idx * BSIZE, batch_size)?;
            let batch_images = batch_images.to_device(&dev)?;
            let batch_labels = batch_labels.to_device(&dev)?;
            let logits = model.forward(&batch_images, false)?;
            let ok = logits
                .argmax(candle_core::D::Minus1)?
                .eq(&batch_labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            test_ok += ok;
            test_total += batch_size;
        }
        let test_accuracy = if test_total > 0 {
            test_ok / test_total as f32
        } else { 0.0 };

        println!(
            "Epoch {epoch:4}: train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_accuracy
        );
    }

    // Calculate validation accuracy after training is complete
    let valid_accuracy = calculate_validation_accuracy(&model, &valid_images, &valid_labels, &dev)?;
    println!("Training complete! Final validation accuracy: {:5.2}%", 100. * valid_accuracy);

    if let Some(save) = &args.save {
        println!("Saving trained weights in {save}");
        varmap.save(save)?;
    }
    Ok(())
}

fn calculate_validation_accuracy(
    model: &ConvNet,
    valid_images: &Tensor,
    valid_labels: &Tensor,
    dev: &candle_core::Device,
) -> anyhow::Result<f32> {
    let mut valid_ok = 0f32;
    let mut valid_total = 0;
    let n_valid = valid_images.dim(0)?;
    let n_valid_batches = (n_valid + BSIZE - 1) / BSIZE;
    
    for batch_idx in 0..n_valid_batches {
        let batch_size = if batch_idx == n_valid_batches - 1 { 
            n_valid - batch_idx * BSIZE 
        } else { 
            BSIZE 
        };
        let batch_images = valid_images.narrow(0, batch_idx * BSIZE, batch_size)?;
        let batch_labels = valid_labels.narrow(0, batch_idx * BSIZE, batch_size)?;
        let batch_images = batch_images.to_device(dev)?;
        let batch_labels = batch_labels.to_device(dev)?;
        let logits = model.forward(&batch_images, false)?;
        let ok = logits
            .argmax(candle_core::D::Minus1)?
            .eq(&batch_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        valid_ok += ok;
        valid_total += batch_size;
    }
    
    let valid_accuracy = if valid_total > 0 {
        valid_ok / valid_total as f32
    } else { 
        0.0 
    };
    
    Ok(valid_accuracy)
}

fn main() -> anyhow::Result<()> {
    let config = Config::from_env();
    config.ensure_directories()?;
    let save_path = Some(config.model_save_path("trained_model.safetensors").to_string_lossy().to_string());
    let load_path = None;

    let data_raw = config.data_dir.join("raw");
    let data_processed = config.data_dir.join("processed");

    // Preprocess only if processed data does not exist
    let processed_ok = ["train", "valid", "test"].iter().all(|split| {
        let dir = data_processed.join(split);
        dir.join("images.npy").exists() && dir.join("labels.npy").exists()
    });
    if !processed_ok {
        println!("Preprocessing face data from {:?} to {:?}", data_raw, data_processed);
        preprocess_and_save(&data_raw, &data_processed)?;
    } else {
        println!("Processed data found in {:?}, skipping preprocessing", data_processed);
    }

    let ((train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels)) = load_processed(&data_processed)?;
    println!("train-images: {:?}", train_images.shape());
    println!("train-labels: {:?}", train_labels.shape());
    println!("valid-images: {:?}", valid_images.shape());
    println!("valid-labels: {:?}", valid_labels.shape());
    println!("test-images: {:?}", test_images.shape());
    println!("test-labels: {:?}", test_labels.shape());

    let training_args = TrainingArgs {
        learning_rate: config.learning_rate,
        epochs: config.epochs,
        save: save_path,
        load: load_path,
    };

    training_loop_cnn(train_images, train_labels, valid_images, valid_labels, test_images, test_labels, &training_args)
}