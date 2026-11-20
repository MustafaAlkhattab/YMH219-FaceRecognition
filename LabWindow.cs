using OpenCvSharp;
using OpenCvSharp.Extensions;
using System.Drawing.Drawing2D;

namespace FaceRecognition;

internal sealed class LabWindow : Form
{
    private static readonly string DemoDirectory = Path.Combine(AppContext.BaseDirectory, "demo");
    private readonly string trainingPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "YMH219-FaceRecognition", "training");
    private readonly PictureBox preview = new() { Dock = DockStyle.Fill, SizeMode = PictureBoxSizeMode.Zoom, BackColor = Color.FromArgb(12, 15, 21), Margin = Padding.Empty };
    private readonly ComboBox label = new() { Width = 198, MaxLength = 40, DropDownStyle = ComboBoxStyle.DropDown };
    private readonly NumericUpDown cameraIndex = new() { Minimum = 0, Maximum = 9, Width = 58 };
    private readonly NumericUpDown threshold = new() { Minimum = 1, Maximum = 200, Value = 72, Width = 64 };
    private readonly Button cameraStart = Ui.Button("Kamera başlat");
    private readonly Button cameraStop = Ui.Button("Durdur");
    private readonly Button photo = Ui.Button("Görüntü seç");
    private readonly Button demo = Ui.Button("Hazır demoyu aç", true);
    private readonly Button next = Ui.Button("Diğer portre");
    private readonly Button useLocal = Ui.Button("Yerel örneklere dön");
    private readonly Button add = Ui.Button("Kareyi örnekle");
    private readonly Button train = Ui.Button("Modeli eğit", true);
    private readonly Label source = Ui.Label("Görüntü bekleniyor", 11, true, Ui.Accent);
    private readonly Label state = Ui.Label("Başlatılıyor", 10, true);
    private readonly Label prediction = Ui.Label("Henüz sonuç yok", 12, true);
    private readonly Label guidance = Ui.Label("", 10, false, Ui.Muted);
    private readonly Label mode = Ui.Label("Algılama · Eğitim yok", 11, true, Ui.Accent);
    private readonly ListBox samples = new() { Dock = DockStyle.Fill, IntegralHeight = false, BorderStyle = BorderStyle.None, Margin = new(0, 6, 0, 8) };
    private readonly System.Windows.Forms.Timer frames = new() { Interval = 125 };
    private FaceLab? lab;
    private VideoCapture? camera;
    private Mat? current;
    private bool demoActive, opening, closed;
    private int demoIndex;
    private static readonly string[] DemoLabels = ["Atlas", "Poyraz"];

    public LabWindow(bool enterDemo)
    {
        Ui.Window(this, "Fırat Face Lab", 1180, 735, 900, 605);
        Ui.Input(label, "Kişi veya sınıf etiketi");
        label.Margin = new(0, 6, 8, 3);
        Ui.Input(cameraIndex, "Kamera numarası");
        cameraIndex.Margin = new(0, 7, 8, 2);
        Ui.Input(threshold, "LBPH uzaklık eşiği");
        threshold.Margin = new(0, 7, 8, 2);
        Ui.Input(samples, "Etiket başına örnek sayıları");
        guidance.MaximumSize = new(260, 0);
        prediction.MaximumSize = new(260, 0);
        state.Margin = new(0, 7, 0, 0);
        source.Margin = new(0, 0, 0, 10);
        var heading = Ui.Label("Fırat Face Lab", 27, true);
        var columns = new TableLayoutPanel { Dock = DockStyle.Fill, Margin = Padding.Empty, ColumnCount = 2, RowCount = 1 };
        columns.ColumnStyles.Add(new(SizeType.Percent, 100));
        columns.ColumnStyles.Add(new(SizeType.Absolute, 312));
        var viewer = Ui.Stack(14, (source, 0), (preview, 100), (prediction, 0));
        viewer.BackColor = Ui.Surface;
        prediction.Margin = new(0, 10, 0, 0);
        var training = Ui.Stack(16,
            (Ui.Label("Örnek kitaplığı", 17, true), 0),
            (mode, 0),
            (Ui.Label("Aynı kişi için aynı etiketi kullanın.", 10, false, Ui.Muted), 0),
            (Ui.Bar(label, add), 0),
            (samples, 100),
            (guidance, 0),
            (Ui.Bar(train), 0),
            (Ui.Bar(ToolCaption("Uzaklık eşiği"), threshold), 0));
        training.BackColor = Ui.Surface;
        training.Margin = new(12, 0, 0, 0);
        columns.Controls.Add(viewer, 0, 0);
        columns.Controls.Add(training, 1, 0);
        var tools = Ui.Bar(ToolCaption("Kamera"), cameraIndex, cameraStart, cameraStop, photo, demo, next, useLocal);
        Controls.Add(Ui.Stack(18, (heading, 0), (tools, 0), (columns, 100), (state, 0)));

        Shown += (_, _) => Run(() =>
        {
            lab = new(Path.Combine(AppContext.BaseDirectory, "models", "haarcascade_frontalface_default.xml"), trainingPath);
            if (enterDemo) StartDemo();
            else { RefreshClasses(); ShowEmpty(); }
        });
        cameraStart.Click += async (_, _) => await StartCamera();
        cameraStop.Click += (_, _) => { StopCamera(); state.Text = "Kamera durdu. Son kareyi örnekleyebilirsiniz."; };
        photo.Click += (_, _) => OpenPhoto();
        demo.Click += (_, _) => Run(StartDemo);
        next.Click += (_, _) => Run(() => ShowDemoPortrait((demoIndex + 1) % DemoLabels.Length));
        useLocal.Click += (_, _) => Run(LeaveDemo);
        add.Click += (_, _) => Run(() =>
        {
            EnsureFrame();
            lab!.AddSample(current!, label.Text);
            RefreshClasses();
            Render();
            state.Text = $"“{label.Text.Trim()}” etiketi için bir yüz örneği eklendi. Güncel sonuç için modeli eğitin.";
        });
        train.Click += (_, _) => Run(() =>
        {
            EnsureLab();
            var classes = lab!.ReadClasses();
            lab.Train(classes);
            if (current is not null) Render();
            mode.Text = classes.Length == 1 ? "Eşleşme · 1 etiket" : $"Tanıma · {classes.Length} etiket";
            guidance.Text = classes.Length == 1
                ? "Tek kişi modeli hazır. Eşik üstündeki yüzler Bilinmiyor olarak gösterilir."
                : $"{classes.Length} etiket ile tanıma etkin.";
            state.Text = $"{classes.Sum(group => group.Paths.Length)} örnek ile eğitim tamamlandı.";
        });
        samples.SelectedIndexChanged += (_, _) =>
        {
            if (!demoActive && samples.SelectedItem is SampleRow row) label.Text = row.Group.Label;
        };
        label.TextChanged += (_, _) => RefreshControls();
        threshold.ValueChanged += (_, _) => Run(() =>
        {
            if (lab is null) return;
            lab.Threshold = (double)threshold.Value;
            if (current is not null) Render();
        });
        frames.Tick += (_, _) => ReadCamera();
        RefreshControls();
    }

    private static Label ToolCaption(string value)
    {
        var title = Ui.Label(value, 10, false, Ui.Muted);
        title.Margin = new(0, 9, 5, 0);
        return title;
    }

    private void RefreshControls()
    {
        bool ready = lab is not null && !opening;
        cameraStart.Enabled = ready && camera is null;
        cameraStop.Enabled = camera is not null;
        cameraIndex.Enabled = camera is null && !opening;
        photo.Enabled = demo.Enabled = ready;
        next.Visible = useLocal.Visible = demoActive;
        label.Enabled = ready && !demoActive;
        add.Enabled = label.Enabled && current is not null && label.Text.Trim().Length > 0;
        train.Enabled = ready && !demoActive && samples.Items.Cast<SampleRow>().Sum(x => x.Group.Paths.Length) >= 2;
    }

    private void RefreshClasses()
    {
        var classes = lab!.ReadClasses();
        samples.Items.Clear();
        samples.Items.AddRange(classes.Select(group => new SampleRow(group)).Cast<object>().ToArray());
        string previous = label.Text;
        label.Items.Clear();
        label.Items.AddRange(classes.Select(group => group.Label).Cast<object>().ToArray());
        label.Text = previous;
        int total = classes.Sum(group => group.Paths.Length);
        if (!lab.Trained) mode.Text = "Algılama · Eğitim yok";
        guidance.Text = total < 2
            ? $"{total}/2 örnek. Bir örnek daha ekleyin. Tek kişiyi denemek için aynı etiketi seçin."
            : $"{classes.Length} etiket · {total} örnek. {FaceLab.RecommendedSamples}+ farklı kare / etiket önerilir.";
        RefreshControls();
    }

    private sealed record SampleRow(SampleClass Group)
    {
        public override string ToString() => $"{Group.Label}  ·  {Group.Paths.Length} örnek";
    }

    private void StartDemo()
    {
        EnsureLab();
        StopCamera();
        var classes = DemoLabels.Select(name => new SampleClass(name,
            [Path.Combine(DemoDirectory, $"{name}-1.jpg"), Path.Combine(DemoDirectory, $"{name}-2.jpg")])).ToArray();
        lab!.Train(classes, portraits: true);
        demoActive = true;
        mode.Text = "Demo · 2 kurgu etiketi";
        samples.Items.Clear();
        samples.Items.AddRange(classes.Select(group => new SampleRow(group)).Cast<object>().ToArray());
        guidance.Text = "Her etiket iki sentetik portreyle eğitildi. Üçüncü portre tahmin için ayrı tutuldu.";
        ShowDemoPortrait(0);
        RefreshControls();
    }

    private void ShowDemoPortrait(int index)
    {
        demoIndex = index;
        SetFrame(FaceLab.LoadPhoto(Path.Combine(DemoDirectory, $"{DemoLabels[index]}-3.jpg")));
        source.Text = "Hazır demo  /  Sentetik portre";
        Render();
        state.Text = $"Test portresi {index + 1}/{DemoLabels.Length} · Beklenen etiket {DemoLabels[index]}";
    }

    private void LeaveDemo()
    {
        if (!demoActive) return;
        demoActive = false;
        lab!.InvalidateModel();
        ClearFrame();
        RefreshClasses();
        ShowEmpty();
    }

    private void OpenPhoto()
    {
        using var picker = new OpenFileDialog { Title = "Bir görüntü seçin", Filter = "Görüntüler|*.jpg;*.jpeg;*.png;*.bmp" };
        if (picker.ShowDialog(this) != DialogResult.OK) return;
        Run(() =>
        {
            EnsureLab();
            StopCamera();
            LeaveDemo();
            SetFrame(FaceLab.LoadPhoto(picker.FileName));
            source.Text = "Dosyadan görüntü";
            state.Text = "Görüntü analizi tamamlandı.";
            Render();
        });
    }

    private async Task StartCamera()
    {
        int index = (int)cameraIndex.Value;
        opening = true;
        RefreshControls();
        state.Text = "Kameraya bağlanılıyor…";
        try
        {
            LeaveDemo();
            state.Text = "Kameraya bağlanılıyor…";
            var capture = await Task.Run(() => OpenCamera(index));
            if (closed) { capture.Dispose(); return; }
            camera = capture;
            frames.Start();
            source.Text = $"Kamera {index}  /  Canlı";
            state.Text = "Kamera açık. Durdur düğmesi son kareyi ekranda bırakır.";
        }
        catch (Exception ex) { if (!closed) ReportError(ex); }
        finally { opening = false; if (!closed) RefreshControls(); }
    }

    private static VideoCapture OpenCamera(int index)
    {
        var capture = new VideoCapture();
        try
        {
            if (!capture.Open(index, VideoCaptureAPIs.DSHOW) && !capture.Open(index, VideoCaptureAPIs.MSMF))
                throw new InvalidOperationException("Kamera açılamadı. Kamera numarasını ve uygulama erişimini kontrol edin.");
            capture.Set(VideoCaptureProperties.FrameWidth, 640);
            capture.Set(VideoCaptureProperties.FrameHeight, 480);
            return capture;
        }
        catch { capture.Dispose(); throw; }
    }

    private void ReadCamera()
    {
        if (camera is null) return;
        Mat? frame = new Mat();
        try
        {
            if (!camera.Read(frame) || frame.Empty()) throw new IOException("Kameradan kare alınamadı.");
            SetFrame(frame);
            frame = null;
            Render();
        }
        catch (Exception ex) { StopCamera(); ReportError(ex); }
        finally { frame?.Dispose(); }
    }

    private void StopCamera()
    {
        frames.Stop();
        camera?.Dispose();
        camera = null;
        if (!closed) RefreshControls();
    }

    private void SetFrame(Mat image)
    {
        if (Math.Max(image.Width, image.Height) > 850)
        {
            double scale = 850d / Math.Max(image.Width, image.Height);
            var resized = new Mat();
            using (image)
                Cv2.Resize(image, resized, new OpenCvSharp.Size(Math.Max(1, (int)(image.Width * scale)), Math.Max(1, (int)(image.Height * scale))));
            image = resized;
        }
        current?.Dispose();
        current = image;
        RefreshControls();
    }

    private void Render()
    {
        EnsureFrame();
        var results = lab!.Analyze(current!);
        var bitmap = BitmapConverter.ToBitmap(current!);
        using (var paint = Graphics.FromImage(bitmap))
        using (var font = new Font("Segoe UI", Math.Max(11, bitmap.Width / 26f), FontStyle.Bold, GraphicsUnit.Pixel))
        {
            paint.SmoothingMode = SmoothingMode.AntiAlias;
            foreach (FaceResult face in results)
            {
                var bounds = face.Bounds;
                using var line = new Pen(face.Label == "Bilinmiyor" ? Ui.Accent : Color.MediumAquamarine, 3);
                paint.DrawRectangle(line, bounds.X, bounds.Y, bounds.Width, bounds.Height);
                string badge = face.Distance is double distance ? $"{face.Label} · {distance:F1}" : face.Label;
                var textSize = paint.MeasureString(badge, font);
                float y = Math.Max(0, bounds.Y - textSize.Height - 5);
                float x = Math.Max(0, Math.Min(bounds.X, bitmap.Width - textSize.Width - 10));
                using var backing = new SolidBrush(Color.FromArgb(225, Ui.Background));
                paint.FillRectangle(backing, x, y, textSize.Width + 10, textSize.Height + 4);
                paint.DrawString(badge, font, Brushes.White, x + 5, y + 2);
            }
        }
        var old = preview.Image;
        preview.Image = bitmap;
        old?.Dispose();
        prediction.Text = results.Length == 0 ? "Yüz bulunamadı"
            : !lab.Trained ? $"{results.Length} yüz  ·  Algılama"
            : string.Join("  |  ", results.Take(3).Select(result => $"{result.Label} · {result.Distance:F1}"))
              + (results.Length > 3 ? $"  +{results.Length - 3}" : "");
    }

    private void ClearFrame()
    {
        current?.Dispose();
        current = null;
        var old = preview.Image;
        preview.Image = null;
        old?.Dispose();
    }

    private void ShowEmpty()
    {
        source.Text = "Görüntü bekleniyor";
        prediction.Text = "Henüz sonuç yok";
        state.Text = "Bir görüntü seçin, kamerayı açın veya hazır demoyu deneyin.";
    }

    private void EnsureLab()
    {
        if (lab is null) throw new InvalidOperationException("Algılama modeli hazır değil.");
    }

    private void EnsureFrame()
    {
        EnsureLab();
        if (current is null) throw new InvalidOperationException("Önce bir görüntü açın veya kameradan kare alın.");
    }

    private void Run(Action action)
    {
        try { action(); }
        catch (Exception ex) { ReportError(ex); }
    }

    private void ReportError(Exception ex) => state.Text = ex.Message;

    protected override void OnFormClosed(FormClosedEventArgs e)
    {
        closed = true;
        StopCamera();
        frames.Dispose();
        ClearFrame();
        lab?.Dispose();
        base.OnFormClosed(e);
    }
}
