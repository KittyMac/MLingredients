import UIKit
import PlanetSwift
import CoreML
import Vision

class HealthModelSource : MLFeatureProvider {
    
    let MAX_LEN = 80
    var ingredientData:MLMultiArray? = nil
    
    func padArray(to numToPad: Int, sequence: [NSNumber]) -> [NSNumber] {
        var newSeq = sequence
        for _ in sequence.count ... numToPad {
            newSeq.insert(NSNumber(value:0.0), at: 0)
        }
        return newSeq
    }
    
    func tokenizer(_ words: [String], _ ingredientWordToIndex:[String:Int]) -> [NSNumber] {
        var tokens : [NSNumber] = []
        for (index, word) in words.enumerated() {
            if let val = ingredientWordToIndex[word] {
                tokens.insert(NSNumber(value: val), at: index)
            } else {
                tokens.insert(NSNumber(value: 0.0), at: index)
            }
        }
        return padArray(to: MAX_LEN-1, sequence: tokens)
    }
    
    init (_ ingredients:String, _ ingredientWordToIndex:[String:Int]) {
        let wordsArray = ingredients.split(separator: " ").map(String.init)
        let inputList = tokenizer(wordsArray, ingredientWordToIndex)
        
        guard let input_data = try? MLMultiArray(shape: [80,1,1], dataType: .double) else {
            fatalError("Unexpected runtime error. MLMultiArray")
        }
        
        for (index,item) in inputList.enumerated() {
            input_data[index] = item
        }
        
        ingredientData = input_data
        
        print(inputList)
    }
    
    public var featureNames : Set<String> {
        get {
            return ["ingredients"]
        }
    }
    
    /// Returns nil if the provided featureName is not in the set of featureNames
    public func featureValue(for featureName: String) -> MLFeatureValue?{
        if featureName == "ingredients" && ingredientData != nil {
            return MLFeatureValue(multiArray: ingredientData!)
        }
        return nil
    }
    
}

class PredictController: SharedController {
    
    var healthModel:MLModel? = nil
    var ocrModel:VNCoreMLModel? = nil
    var ingredientWordToIndex = [String:Int]()
    
    func loadModels() {
        do {
            // The OCR model converts image of character into a character
            let ocrModelURL = URL(fileURLWithPath: String(bundlePath:"bundle://Assets/predict/ocr.mlmodel"))
            let ocrCompiledUrl = try MLModel.compileModel(at: ocrModelURL)
            let ocrModel = try MLModel(contentsOf: ocrCompiledUrl)
            self.ocrModel = try? VNCoreMLModel(for: ocrModel)
            
            // the health model converts an ingredient string to a health score
            let healthModelURL = URL(fileURLWithPath: String(bundlePath:"bundle://Assets/predict/ingredients.mlmodel"))
            let healthCompiledUrl = try MLModel.compileModel(at: healthModelURL)
            self.healthModel = try MLModel(contentsOf: healthCompiledUrl)
            
            
            // load in the ingredient word to index look up table
            do {
                let ingredientsWordListURL = URL(fileURLWithPath: String(bundlePath:"bundle://Assets/predict/words.txt"))
                let ingredientsWordListString = try String(contentsOf: ingredientsWordListURL, encoding: .utf8)
                let ingredientsWordList = ingredientsWordListString.split(separator: "\n")
                
                var idx = 1
                for word in ingredientsWordList {
                    ingredientWordToIndex[String(word)] = idx
                    idx += 1
                }
            }
            catch {/* error handling here */}
            
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
    
        finalString = finalString.replacingOccurrences(of: " ", with: "").lowercased()
        finalString = finalString.replacingOccurrences(of: ",", with: " ").lowercased()
        print(finalString)
        ocrResults.label.text = finalString
        
        
        // Clear up the OCR results
        // 1. Check each ingredients against our ingredients dictionary, if its close enough assume that is the word
        
        
        
        // Predict against the ingredients model to discover the health score
        if let healthModel = self.healthModel {
            do {
                let source = HealthModelSource(finalString, ingredientWordToIndex)
                let prediction = try healthModel.prediction(from: source)
                if let result = prediction.featureValue(for: "health") {
                    predictionResults.label.text = String(format:"Health Factor: %d", Int(result.multiArrayValue![0].doubleValue * 100.0))
                }
                
            } catch {
                print(error)
            }
        }
        
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
    fileprivate var ocrResults: Label {
        return mainXmlView!.elementForId("ocrResults")!.asLabel!
    }
    fileprivate var predictionResults: Label {
        return mainXmlView!.elementForId("predictionResults")!.asLabel!
    }
    fileprivate var buttonsContainer: View {
        return mainXmlView!.elementForId("buttonsContainer")!.asView!
    }
}

