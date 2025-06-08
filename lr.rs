use ndarray::{Array2, Array1};

fn main() {
    let x  = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0
        ];
    let y = vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let x: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> = Array2::from_shape_vec((6, 2), x).unwrap();
    let y: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = Array1::from(y);
    
    
    let mut clf = LR::new(x, y);
    clf.lr = 0.01;
    for i in 0..10000 {
        clf.predict();
        clf.grad();
        clf.update()
    }
    println!("w: {:?}, b: {}", clf.w, clf.b);
    println!("r2: {}, loss: {}", clf.r2score(), clf.mse());
}


#[derive(Debug, Clone)]
struct LR {
    x: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
    y: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>,
    b: f64,
    w: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>,
    lr: f64,
    m: f64,
    pred: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>,
    wrt_w: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>,
    wrt_b: f64,
}

impl LR {
    fn new(x: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>, y: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>) -> Self {
        LR {
            x: x.clone(),
            y: y,
            b: 0.0,
            w: Array1::<f64>::zeros(x.shape()[1]),
            lr: 0.01,
            m: x.shape()[0] as f64,
            pred: Array1::from_vec(Vec::new()),
            wrt_w: Array1::from_vec(Vec::new()),
            wrt_b: 0.0,
        }
    }

    fn predict(&mut self) {
        self.pred = self.x.dot(&self.w) + self.b;
    }

    fn grad(&mut self) {
        let err = &self.pred - &self.y;
        self.wrt_w = (2.0 / self.m) * self.x.t().dot(&err);
        self.wrt_b = (2.0 / self.m) * err.sum();
    }   

    fn mse(&self) -> f64{
        self.pred.iter().zip(self.y.iter()).map(|(p, y)| (p - y).powi(2)).sum()
    }

    fn update(&mut self) {
        self.w = self.w.clone() - self.lr * self.wrt_w.clone();
        self.b = self.b - self.lr * self.wrt_b;
    }

    fn r2score(&self) -> f64 {
        let r2 = 1.0;
        let mut rss = 0.0;
        let mut tss = 0.0;
        for i in 0..self.y.len(){
            rss += (self.y[i] - self.pred[i]).powf(2.0);
        }

        for i in 0..self.y.len(){
            tss += (self.y[i] - self.y.mean().unwrap()).powf(2.0);
        }

        r2 - rss/tss
    }
}
