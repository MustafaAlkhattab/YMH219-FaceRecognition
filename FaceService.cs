using OpenCvSharp;
using OpenCvSharp.Face;
namespace FaceRecognition;

public sealed record FaceResult(Rect Bounds, string Label, double? Distance);

public sealed class FaceService : IDisposable
{
    private readonly CascadeClassifier detector;
    private LBPHFaceRecognizer? recognizer;
    private readonly Dictionary<int, string> labels = [];
    public string TrainingPath { get; } = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "YMH219-FaceRecognition", "training");
    public bool Trained => recognizer is not null;
    public double Threshold { get; set; } = 70;
    public FaceService(string modelPath)
    {
        if (!File.Exists(modelPath)) throw new FileNotFoundException("Yüz algılama modeli bulunamadı. models klasörünü kontrol edin.", modelPath);
        detector = new(modelPath); if (detector.Empty()) throw new InvalidDataException("Yüz algılama modeli yüklenemedi.");
        Directory.CreateDirectory(TrainingPath);
    }
    public Rect[] Detect(Mat image)
    {
        using var gray = Gray(image);
        return detector.DetectMultiScale(gray, 1.1, 5, HaarDetectionTypes.ScaleImage, new OpenCvSharp.Size(60, 60));
    }
    private static Mat Gray(Mat image)
    {
        var gray = new Mat(); if (image.Channels() == 1) image.CopyTo(gray); else Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
        Cv2.EqualizeHist(gray, gray); return gray;
    }
    private static Mat Crop(Mat image, Rect rectangle)
    {
        using var gray = Gray(image); using var region = new Mat(gray, rectangle);
        var face = new Mat(); Cv2.Resize(region, face, new OpenCvSharp.Size(160, 160)); return face;
    }
    public string AddSample(Mat image, string label)
    {
        label = label.Trim();
        if (label.Length is < 1 or > 40 || label is "." or ".." || label.Any(c => !char.IsLetterOrDigit(c) && c != '-' && c != '_')) throw new ArgumentException("Etiket 1-40 harf, rakam, - veya _ içermeli.");
        var faces = Detect(image); if (faces.Length != 1) throw new InvalidOperationException($"Örnek eklemek için görüntüde tam bir yüz olmalı. Bulunan: {faces.Length}");
        var directory = Path.Combine(TrainingPath, label); Directory.CreateDirectory(directory);
        using var crop = Crop(image, faces[0]); var path = Path.Combine(directory, Guid.NewGuid() + ".png");
        if (!Cv2.ImWrite(path, crop)) throw new IOException("Örnek kaydedilemedi.");
        // Yeni örnek eklendikten sonra eski eğitim sonucu kullanılmaz.
        recognizer?.Dispose(); recognizer = null; return path;
    }
    public string[] SampleSummary() => Directory.GetDirectories(TrainingPath).Select(d => $"{Path.GetFileName(d)}: {Directory.GetFiles(d, "*.png").Length} örnek").ToArray();
    public void Train()
    {
        var images = new List<Mat>(); var ids = new List<int>(); var nextLabels = new Dictionary<int, string>();
        try
        {
            foreach (var dir in Directory.GetDirectories(TrainingPath))
            {
                var files = Directory.GetFiles(dir, "*.png"); if (files.Length < 2) continue;
                var id = nextLabels.Count; nextLabels[id] = Path.GetFileName(dir);
                foreach (var file in files)
                {
                    using var source = Cv2.ImRead(file, ImreadModes.Grayscale); if (source.Empty()) throw new InvalidDataException("Örnek okunamadı: " + file);
                    var normalized = new Mat(); Cv2.Resize(source, normalized, new OpenCvSharp.Size(160, 160)); images.Add(normalized); ids.Add(id);
                }
            }
            if (nextLabels.Count < 2) throw new InvalidOperationException("En az iki etiket ve her etikette en az iki yüz örneği ekleyin.");
            var next = LBPHFaceRecognizer.Create();
            try { next.Train(images, ids); } catch { next.Dispose(); throw; }
            recognizer?.Dispose(); recognizer = next; labels.Clear(); foreach (var pair in nextLabels) labels[pair.Key] = pair.Value;
        }
        finally { foreach (var image in images) image.Dispose(); }
    }
    public FaceResult[] Analyze(Mat image)
    {
        return Detect(image).Select(rect =>
        {
            if (recognizer is null) return new FaceResult(rect, "Eğitim yok", null);
            using var face = Crop(image, rect); recognizer.Predict(face, out int id, out double distance);
            return new FaceResult(rect, distance <= Threshold && labels.TryGetValue(id, out var label) ? label : "Bilinmiyor", distance);
        }).ToArray();
    }
    public void Dispose() { recognizer?.Dispose(); detector.Dispose(); }
}
