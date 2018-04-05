import UIKit
import PlanetSwift
import CoreML
import Vision

class PredictController: SharedController {
    
    var model:VNCoreMLModel? = nil
    var ocrModel:VNCoreMLModel? = nil
    
    func loadModels() {
        do {
            // The OCR model converts image of character into a character
            let modelURL = URL(fileURLWithPath: String(bundlePath:"bundle://Assets/predict/ocr.mlmodel"))
            let compiledUrl = try MLModel.compileModel(at: modelURL)
            let model = try MLModel(contentsOf: compiledUrl)
            self.ocrModel = try? VNCoreMLModel(for: model)
        } catch {
            print(error)
        }
    }
    
    override func newObservationsAvailable () {
        self.preview.imageView.image = UIImage(ciImage: lastOriginalImage!)
        
        // run through all obersvations and process them immediately, showing the OCR'd results
        var finalString = ""
        
        for perspectiveImagesCoords in self.observationsToProcess {
            
            if perspectiveImagesCoords == nil {
                finalString += " "
                continue
            }
        
            let extractedImage = self.lastOriginalImage!.applyingFilter("CIPerspectiveCorrection", parameters: perspectiveImagesCoords as! [String : Any])
            let scaledImage = extractedImage.transformed(by: CGAffineTransform.init(scaleX: 28.0 / extractedImage.extent.size.width, y: 28.0 / extractedImage.extent.size.height))
            
            let handler = VNImageRequestHandler(ciImage: scaledImage)
            do {
                let request = VNCoreMLRequest(model: self.ocrModel!)
                try handler.perform([request])
                guard let results = request.results as? [VNClassificationObservation] else {
                    return
                }
                if results[0].confidence > 0.0 {
                    finalString += results[0].identifier
                }
                
            } catch {
                print(error)
            }
        }
            
        print(finalString)
        
    }
    
    override func viewDidLoad() {
        title = "Predict"
        mainBundlePath = "bundle://Assets/predict/predict.xml"
        loadView()
        
        super.viewDidLoad()
        
        UIApplication.shared.isIdleTimerDisabled = true
        
        loadModels()
        
        // debug to save the current image
        let tap = UITapGestureRecognizer(target: self, action: #selector(self.SaveImages(_:)))
        preview.view.addGestureRecognizer(tap)
        preview.view.isUserInteractionEnabled = true
        
    }
    
    @objc func SaveImages(_ sender: UITapGestureRecognizer) {
        let originalImage = ciContext.createCGImage(lastOriginalImage!, from: lastOriginalImage!.extent)
        UIImageWriteToSavedPhotosAlbum(UIImage(cgImage: originalImage!), self, #selector(self.image(_:didFinishSavingWithError:contextInfo:)), nil)
    }
    
    @objc func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            // we got back an error!
            let ac = UIAlertController(title: "Save error", message: error.localizedDescription, preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        } else {
            let ac = UIAlertController(title: "Saved!", message: "Your image has been saved to your photos.", preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        }
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        UIApplication.shared.isIdleTimerDisabled = false
        super.viewDidDisappear(animated)
    }
    
    fileprivate var preview: ImageView {
        return mainXmlView!.elementForId("preview")!.asImageView!
    }
    fileprivate var nextObservation: Button {
        return mainXmlView!.elementForId("nextObservation")!.asButton!
    }
    fileprivate var predictionLabel: Label {
        return mainXmlView!.elementForId("predictionLabel")!.asLabel!
    }
    fileprivate var buttonsContainer: View {
        return mainXmlView!.elementForId("buttonsContainer")!.asView!
    }
}

