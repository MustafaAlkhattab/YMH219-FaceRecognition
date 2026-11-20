using OpenCvSharp;
using OpenCvSharp.Face;

namespace FaceRecognition;

internal sealed record FaceResult(Rect Bounds, string Label, double? Distance);
internal sealed record SampleClass(string Label, string[] Paths);

internal sealed class FaceLab : IDisposable
{
    public const int RecommendedSamples = 3;
    private readonly CascadeClassifier detector;
    private LBPHFaceRecognizer? recognizer;
    private string[] labels = [];
    private readonly string trainingPath;
    public bool Trained => recognizer is not null;
    public double Threshold { get; set; } = 72;

    public FaceLab(string modelPath, string trainingPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("Haar modeli bulunamadı. Uygulamayı models klasörüyle birlikte açın.", modelPath);
        detector = new(modelPath);
        if (detector.Empty())
        {
            detector.Dispose();
            throw new InvalidDataException("Haar modeli yüklenemedi.");
        }
        this.trainingPath = trainingPath;
    }

    public Rect[] Detect(Mat image)
    {
        using var gray = Gray(image);
        return detector.DetectMultiScale(gray, 1.10, 5, minSize: new OpenCvSharp.Size(46, 46));
    }

    private static Mat Gray(Mat image)
    {
        var gray = new Mat();
        if (image.Channels() == 1) image.CopyTo(gray);
        else Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
        return gray;
    }

    private static Mat Normalize(Mat image, Rect area)
    {
        using var gray = Gray(image);
        using var region = new Mat(gray, area);
        var face = new Mat();
        Cv2.Resize(region, face, new OpenCvSharp.Size(160, 160));
        Cv2.EqualizeHist(face, face);
        return face;
    }

    public SampleClass[] ReadClasses()
    {
        if (!Directory.Exists(trainingPath)) return [];
        return Directory.GetDirectories(trainingPath)
            .OrderBy(Path.GetFileName, StringComparer.CurrentCultureIgnoreCase)
            .Select(directory => new SampleClass(Path.GetFileName(directory), Directory.GetFiles(directory, "*.png")))
            .Where(group => group.Paths.Length != 0)
            .ToArray();
    }

    public void AddSample(Mat image, string label)
    {
        label = label.Trim();
        if (label.Length is < 1 or > 40 || label.Any(c => !char.IsLetterOrDigit(c) && c is not ('-' or '_')))
            throw new ArgumentException("Etiket 1–40 harf, rakam, tire veya alt çizgi içermeli.");
        var found = Detect(image);
        if (found.Length != 1)
            throw new InvalidOperationException($"Örnek eklemek için karede tam bir yüz gerekli. Bulunan yüz: {found.Length}.");
        var directory = Path.Combine(trainingPath, label);
        Directory.CreateDirectory(directory);
        using var cropped = Normalize(image, found[0]);
        if (!Cv2.ImEncode(".png", cropped, out byte[] bytes))
            throw new IOException("Örnek dosyası yazılamadı.");
        File.WriteAllBytes(Path.Combine(directory, Guid.NewGuid().ToString("N") + ".png"), bytes);
        InvalidateModel();
    }

    public static bool CanTrain(IReadOnlyList<SampleClass> classes) => classes.Sum(group => group.Paths.Length) >= 2;

    public void Train(IReadOnlyList<SampleClass> classes, bool portraits = false)
    {
        if (!CanTrain(classes))
            throw new InvalidOperationException("Eğitim için en az iki yüz örneği ekleyin. Aynı kişinin fotoğrafları tek etikette olmalı.");

        var images = new List<Mat>();
        var ids = new List<int>();
        try
        {
            for (int id = 0; id < classes.Count; id++)
            {
                foreach (var path in classes[id].Paths)
                {
                    using var image = LoadPhoto(path);
                    if (portraits)
                    {
                        Rect[] found = Detect(image);
                        if (found.Length != 1) throw new InvalidDataException("Demo örneğinde tek bir yüz bulunamadı.");
                        images.Add(Normalize(image, found[0]));
                    }
                    else
                    {
                        // Existing samples are already cropped and equalized.
                        using var gray = Gray(image);
                        var size = new OpenCvSharp.Size(160, 160);
                        var resized = new Mat();
                        Cv2.Resize(gray, resized, size);
                        images.Add(resized);
                    }
                    ids.Add(id);
                }
            }
            var next = LBPHFaceRecognizer.Create();
            try { next.Train(images, ids); }
            catch { next.Dispose(); throw; }
            recognizer?.Dispose();
            recognizer = next;
            labels = classes.Select(group => group.Label).ToArray();
        }
        finally
        {
            foreach (Mat image in images) image.Dispose();
        }
    }

    public static Mat LoadPhoto(string path)
    {
        using var stream = File.OpenRead(path);
        if (stream.Length > 20 * 1024 * 1024)
            throw new ArgumentException("En fazla 20 MB boyutunda bir fotoğraf seçin.");
        var bytes = new byte[checked((int)stream.Length)];
        stream.ReadExactly(bytes);
        var image = Cv2.ImDecode(bytes, ImreadModes.Color);
        if (!image.Empty()) return image;
        image.Dispose();
        throw new InvalidDataException("Görüntü okunamadı. JPG, PNG veya BMP dosyası seçin.");
    }

    public FaceResult[] Analyze(Mat image)
    {
        var results = new List<FaceResult>();
        foreach (Rect rect in Detect(image))
        {
            if (recognizer is null)
            {
                results.Add(new(rect, "Yüz algılandı", null));
                continue;
            }
            using var face = Normalize(image, rect);
            recognizer.Predict(face, out int id, out double distance);
            var label = id >= 0 && id < labels.Length && distance <= Threshold ? labels[id] : "Bilinmiyor";
            results.Add(new(rect, label, distance));
        }
        return results.ToArray();
    }

    public void InvalidateModel()
    {
        recognizer?.Dispose();
        recognizer = null;
        labels = [];
    }

    public void Dispose()
    {
        InvalidateModel();
        detector.Dispose();
    }
}
