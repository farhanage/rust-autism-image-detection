use std::env;
use std::path::PathBuf;

pub struct Config {
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
    pub learning_rate: f64,
    pub epochs: usize,
    #[allow(dead_code)]
    pub model_type: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            data_dir: PathBuf::from(env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string())),
            models_dir: PathBuf::from(env::var("MODELS_DIR").unwrap_or_else(|_| "./models".to_string())),
            learning_rate: env::var("LEARNING_RATE")
                .unwrap_or_else(|_| "0.001".to_string())
                .parse()
                .unwrap_or(0.001),
            epochs: env::var("EPOCHS")
                .unwrap_or_else(|_| "10".to_string())
                .parse()
                .unwrap_or(10),
            model_type: env::var("MODEL_TYPE").unwrap_or_else(|_| "Cnn".to_string()),
        }
    }

    pub fn ensure_directories(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(&self.models_dir)?;
        std::fs::create_dir_all(self.data_dir.join("raw"))?;
        std::fs::create_dir_all(self.data_dir.join("processed"))?;
        Ok(())
    }

    pub fn model_save_path(&self, filename: &str) -> PathBuf {
        self.models_dir.join(filename)
    }
}
