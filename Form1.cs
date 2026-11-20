using OpenCvSharp;
using OpenCvSharp.Extensions;
namespace FaceRecognition;

public partial class Form1 : Form
{
    private FaceService? faces;
    private VideoCapture? camera;
    private Mat? current;
    private readonly System.Windows.Forms.Timer timer = new() { Interval = 100 };
    private readonly PictureBox preview = new() { Dock = DockStyle.Fill, SizeMode = PictureBoxSizeMode.Zoom, BackColor = Color.FromArgb(20, 26, 43) };
    private readonly ListBox samples = new() { Dock = DockStyle.Fill };
    private readonly Label status = new() { AutoSize = true, Text = "Görüntü seçin veya kamerayı başlatın.", ForeColor = Color.DarkBlue };
    private readonly TextBox label = new() { Width = 180, PlaceholderText = "Örnek: Mustafa_1", AccessibleName = "Yerel örnek etiketi" };
    private readonly NumericUpDown cameraIndex = new() { Minimum = 0, Maximum = 9, Width = 55 };
    private readonly NumericUpDown threshold = new() { Minimum = 1, Maximum = 200, Value = 70, Width = 65 };
    private readonly Button start = new() { Text = "Kamera başlat", AutoSize = true };
    private readonly Button stop = new() { Text = "Kamera durdur", AutoSize = true, Enabled = false };
    private readonly Label modelStatus = new() { Text = "Model: eğitim yok", AutoSize = true };
    public Form1()
    {
        InitializeComponent(); Controls.Clear(); AutoScaleMode = AutoScaleMode.None;
        Text = "Fırat Face Lab | YMH219"; ClientSize = new(1180, 850); MinimumSize = new(1000, 750); StartPosition = FormStartPosition.CenterScreen; Font = new("Segoe UI", 10); BackColor = Color.FromArgb(250, 245, 255);
        var root = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new(24), RowCount = 7 };
        root.RowStyles.Add(new(SizeType.Absolute, 60)); root.RowStyles.Add(new(SizeType.Absolute, 55)); root.RowStyles.Add(new(SizeType.Absolute, 90)); root.RowStyles.Add(new(SizeType.Percent, 100)); root.RowStyles.Add(new(SizeType.Absolute, 95)); root.RowStyles.Add(new(SizeType.Absolute, 60)); root.RowStyles.Add(new(SizeType.Absolute, 30));
        root.Controls.Add(new Label { Text = "FIRAT FACE LAB", AutoSize = true, Font = new("Segoe UI", 24, FontStyle.Bold), ForeColor = Color.FromArgb(109, 40, 217) }, 0, 0);
        root.Controls.Add(new Label { Text = "Yerel yüz algılama ve örneklerle tanıma.\nYalnızca kendi yüzünüzü veya onay veren gönüllülerin örneklerini kullanın.", AutoSize = true }, 0, 1);
        var tools = new FlowLayoutPanel { Dock = DockStyle.Fill }; var select = new Button { Text = "Görüntü seç", AutoSize = true };
        tools.Controls.AddRange([new Label { Text = "Kamera", AutoSize = true }, cameraIndex, start, stop, select, new Label { Text = "LBPH eşik", AutoSize = true }, threshold]); root.Controls.Add(tools, 0, 2);
        var body = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2 }; body.ColumnStyles.Add(new(SizeType.Percent, 75)); body.ColumnStyles.Add(new(SizeType.Percent, 25)); body.Controls.Add(preview); body.Controls.Add(samples); root.Controls.Add(body, 0, 3);
        var training = new FlowLayoutPanel { Dock = DockStyle.Fill, Padding = new(0, 12, 0, 0) }; var add = new Button { Text = "Yüz örneği ekle", AutoSize = true }; var train = new Button { Text = "Modeli eğit", AutoSize = true }; var refresh = new Button { Text = "Yeniden analiz", AutoSize = true };
        training.Controls.AddRange([new Label { Text = "Etiket", AutoSize = true }, label, add, train, refresh, modelStatus]); root.Controls.Add(training, 0, 4);
        root.Controls.Add(status, 0, 5); root.Controls.Add(new Label { Text = "Mustafa Alkhatta · 250541607 | Örnekler yalnızca bu bilgisayarda tutulur.", AutoSize = true }, 0, 6); Controls.Add(root);
        Shown += (_, _) => Run(() => { faces = new(Path.Combine(AppContext.BaseDirectory, "models", "haarcascade_frontalface_default.xml")); RefreshSamples(); status.Text = "Algılama modeli hazır. Önce bir görüntü seçin."; });
        select.Click += (_, _) => { StopCamera(); using var dialog = new OpenFileDialog { Filter = "Görüntüler|*.png;*.jpg;*.jpeg;*.bmp" }; if (dialog.ShowDialog(this) == DialogResult.OK) Run(() => { var image = Cv2.ImRead(dialog.FileName); if (image.Empty()) { image.Dispose(); throw new InvalidDataException("Görüntü okunamadı."); } SetCurrent(image); Analyze(); }); };
        start.Click += (_, _) => Run(() => { StopCamera(); camera = new((int)cameraIndex.Value, VideoCaptureAPIs.DSHOW); if (!camera.IsOpened()) { StopCamera(); throw new InvalidOperationException("Kamera açılamadı. Kamera numarasını, cihaz bağlantısını ve erişim iznini kontrol edin."); } start.Enabled = cameraIndex.Enabled = false; stop.Enabled = true; timer.Start(); status.Text = "Kamera aktif."; });
        stop.Click += (_, _) => { StopCamera(); status.Text = "Kamera durduruldu."; };
        timer.Tick += (_, _) => Run(() => { var frame = new Mat(); if (camera is null || !camera.Read(frame) || frame.Empty()) { frame.Dispose(); StopCamera(); throw new IOException("Kameradan görüntü alınamadı."); } SetCurrent(frame); Analyze(); });
        add.Click += (_, _) => Run(() => { RequireImage(); faces!.AddSample(current!, label.Text); RefreshSamples(); Analyze(); status.Text = "Yerel yüz örneği eklendi. İki etiket için örnek ekleyip modeli eğitin."; });
        train.Click += (_, _) => Run(() => { RequireService(); faces!.Train(); RefreshSamples(); if (current is not null) Analyze(); status.Text = "LBPH modeli yerel örneklerle eğitildi."; });
        refresh.Click += (_, _) => Run(Analyze); threshold.ValueChanged += (_, _) => { if (faces is not null) faces.Threshold = (double)threshold.Value; };
    }
    private void RequireService() { if (faces is null) throw new InvalidOperationException("Algılama modeli yüklenmedi. models klasörünü kontrol edin."); }
    private void RequireImage() { RequireService(); if (current is null || current.Empty()) throw new InvalidOperationException("Önce görüntü seçin veya kamerayı başlatın."); }
    private void SetCurrent(Mat image)
    {
        if (image.Width > 960) { var small = new Mat(); Cv2.Resize(image, small, new OpenCvSharp.Size(960, image.Height * 960 / image.Width)); image.Dispose(); image = small; }
        current?.Dispose(); current = image;
    }
    private void Analyze()
    {
        RequireImage(); var results = faces!.Analyze(current!); var bitmap = BitmapConverter.ToBitmap(current!);
        using (var graphics = Graphics.FromImage(bitmap))
        using (var pen = new Pen(Color.LimeGreen, 3))
        using (var font = new Font("Segoe UI", 12, FontStyle.Bold))
        {
            foreach (var result in results)
            {
                var r = result.Bounds; graphics.DrawRectangle(pen, r.X, r.Y, r.Width, r.Height);
                var name = result.Label + (result.Distance is double distance ? $" · uzaklık {distance:F1}" : "");
                var size = graphics.MeasureString(name, font); var y = Math.Max(0, r.Y - size.Height);
                graphics.FillRectangle(Brushes.Black, r.X, y, size.Width, size.Height); graphics.DrawString(name, font, Brushes.White, r.X, y);
            }
        }
        var old = preview.Image; preview.Image = bitmap; old?.Dispose();
        status.Text = $"{(camera is null ? "Görüntü" : "Kamera")} · {results.Length} yüz · " + (faces.Trained ? "LBPH aktif; düşük uzaklık daha yakın eşleşme." : "Algılama aktif; tanıma için iki etiketle eğitim gerekli.");
        modelStatus.Text = faces.Trained ? "Model: eğitildi" : "Model: eğitim yok";
    }
    private void RefreshSamples() { samples.Items.Clear(); samples.Items.AddRange(faces!.SampleSummary()); modelStatus.Text = faces.Trained ? "Model: eğitildi" : "Model: eğitim yok"; }
    private void StopCamera() { timer.Stop(); camera?.Dispose(); camera = null; start.Enabled = cameraIndex.Enabled = true; stop.Enabled = false; }
    private void Run(Action action) { try { action(); } catch (Exception ex) { status.Text = "Hata: " + ex.Message; MessageBox.Show(this, ex.Message, "Görüntü / model hatası", MessageBoxButtons.OK, MessageBoxIcon.Warning); } }
    protected override void OnFormClosed(FormClosedEventArgs e) { StopCamera(); timer.Dispose(); current?.Dispose(); preview.Image?.Dispose(); faces?.Dispose(); base.OnFormClosed(e); }
}
