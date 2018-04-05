import UIKit
import PlanetSwift
import CoreML
import Vision

class TrainController: SharedController {
    
    var model:VNCoreMLModel? = nil
    var ocrModel:VNCoreMLModel? = nil
    
    let trainingImagesPublisher:SwiftyZeroMQ.Socket? = Comm.shared.publisher(Comm.endpoints.pub_TrainingImages)
    
    func showCurrentObservation() {
        
        //print(self.observationsToProcess)
        
        if observationsToProcess.count == 0 {
            clearObservations()
            return
        }
        
        let perspectiveImagesCoords = self.observationsToProcess[0]
        if perspectiveImagesCoords == nil {
            goToNextObservation()
            return
        }
        
        DispatchQueue.main.async {
            let extractedImage = self.lastOriginalImage!.applyingFilter("CIPerspectiveCorrection", parameters: perspectiveImagesCoords as! [String : Any])
            let scaledImage = extractedImage.transformed(by: CGAffineTransform.init(scaleX: 28.0 / extractedImage.extent.size.width, y: 28.0 / extractedImage.extent.size.height))
            self.preview.imageView.image = UIImage(ciImage: scaledImage)
            
            
            let handler = VNImageRequestHandler(ciImage: scaledImage)
            do {
                let request = VNCoreMLRequest(model: self.ocrModel!)
                try handler.perform([request])
                guard let results = request.results as? [VNClassificationObservation] else {
                    return
                }
                
                self.predictionLabel.label.text = "\"\(results[0].identifier)\" \(Int(results[0].confidence * 10.0))"
                
            } catch {
                print(error)
            }
            
        }
    }
    
    func goToNextObservation() {
        if observationsToProcess.count > 0 {
            observationsToProcess.remove(at: 0)
        }
        showCurrentObservation()
    }
    
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
        showCurrentObservation()
    }
    
    override func viewDidLoad() {
        title = "Train"
        mainBundlePath = "bundle://Assets/train/train.xml"
        loadView()
        
        super.viewDidLoad()
        
        UIApplication.shared.isIdleTimerDisabled = true
        
        loadModels()
        
        // debug to save the current image
        let tap = UITapGestureRecognizer(target: self, action: #selector(self.SaveImages(_:)))
        preview.view.addGestureRecognizer(tap)
        preview.view.isUserInteractionEnabled = true
        
        nextObservation.button.add(for: .touchUpInside) {
            self.goToNextObservation()
        }
        
        // Generate all of the class buttons we need and space them nicely in the buttonsContainer
        for i in 0..<ALPHABET.count {
            _ = CreateButton(UInt8(i), ALPHABET[i...i])
        }
        
    }
    
    func CreateButton(_ classValue:UInt8, _ label:String) -> Button {
        
        var leftMargin = 5
        let topMargin = 5
        
        let sizeOfButton = 58
        let buttonPadding = 4
        let buttonsPerRow = (Int(view.frame.width) - leftMargin * 2) / sizeOfButton
        let btnIdx = buttonsContainer.view.subviews.count
        let xIdx = btnIdx % buttonsPerRow
        let yIdx = btnIdx / buttonsPerRow
        
        leftMargin += (Int(view.frame.width) - buttonsPerRow * sizeOfButton) / 2
        
        let btn = Button().new("ButtonStd") { this in
            this.title = label
            
            this.frame = CGRect(x: buttonPadding + leftMargin + xIdx * sizeOfButton, y: buttonPadding + topMargin + yIdx * sizeOfButton, width: sizeOfButton - buttonPadding * 2, height: sizeOfButton - buttonPadding * 2)
        }
        
        btn.button.add(for: .touchUpInside) {
            // submit the image and correct labelling to the server
            var filename = String.init(format: "%@", UUID().uuidString)
            
            for i in 0..<self.ALPHABET.count {
                filename += String.init(format: "_%d", (btnIdx == i ? 1 : 0))
            }
            
            filename += ".png"
            
            let extractedImage = self.lastOriginalImage!.applyingFilter("CIPerspectiveCorrection", parameters: self.observationsToProcess[0] as! [String : Any])
            let scaledImage = extractedImage.transformed(by: CGAffineTransform.init(scaleX: 28.0 / extractedImage.extent.size.width, y: 28.0 / extractedImage.extent.size.height))
            guard let pngData = self.ciContext.pngRepresentation(of:scaledImage, format: kCIFormatARGB8, colorSpace: CGColorSpaceCreateDeviceRGB(), options: [:]) else {
                return
            }
            
            var dataPacket = Data()
            dataPacket.append(contentsOf: filename.asciiArray8 as [UInt8])
            dataPacket.append(0)
            dataPacket.append(pngData)
            try! self.trainingImagesPublisher?.send(data: dataPacket)
            
            self.goToNextObservation()
        }

        buttonsContainer.view.addSubview(btn.button)
        
        return btn
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

