
using OpenCvSharp;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace KatImageDetector
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {

        public class ImageCard : INotifyPropertyChanged
        {
            public string? Title { get; set; }
            public DetectorType Type { get; set; }
            public Collection<ImageInfo> ImagenesInfo { get; set; } = [];

            public int _currentIndex = 0;

            public ImageInfo? CurrentImage {
                get => ImagenesInfo.Count == 0 ? null : ImagenesInfo [_currentIndex];
                set {                     
                    if (value == null) return;
                    var index = ImagenesInfo.IndexOf(value);
                    if (index != -1)
                    {
                        _currentIndex = index;
                        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(CurrentImage)));
                    }
                }
            }
            public void Update2NextImage()
            {
                Debug.WriteLine("Updated");
                if (ImagenesInfo.Count == 0) return;
                _currentIndex = (_currentIndex + 1) % ImagenesInfo.Count;

                //avisar a la UI que cambió la imagen actual (si usas INotifyPropertyChanged, dispara el evento aquí)
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(CurrentImage)));

            }

            public event PropertyChangedEventHandler? PropertyChanged;
        }

        public class ImageInfo
        {
            public required BitmapSource Image { get; set; }
            public required string TextoInfo { get; set; }
            public string Label { get; set; } = "img";
        }

        public ObservableCollection<ImageCard> Cards { get; set; } = new();

        public MainWindow()
        {
            InitializeComponent();
            DataContext = this;
        }

        private async void BtnOnOff_Click(object sender, RoutedEventArgs e)
        {
            btnIniciar.IsEnabled = false;
            btnIniciar.Content = "Capturando...";

            await EscanearPantalla();

            btnIniciar.IsEnabled = true;
            btnIniciar.Content = "Capturar";
        }

        private void Card_MouseDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            Debug.WriteLine("Card Clicked");
            if (sender is Border border && border.DataContext is ImageCard card)
            {
                card.Update2NextImage();
            }
        }


        private async Task EscanearPantalla()
        {
            // Cargando Imagenes
            using var captura = new OpenCvSharp.Mat(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "pantallazo.png"), OpenCvSharp.ImreadModes.Grayscale);
            using var objetivo = new OpenCvSharp.Mat(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "boton_objetivo.png"), OpenCvSharp.ImreadModes.Grayscale);

            // Modificaciones
            using var capturaMod = new OpenCvSharp.Mat(captura, new OpenCvSharp.Rect(0, 0, captura.Width / 4, captura.Height / 4));
            using var objetivoMod = new OpenCvSharp.Mat();
            Cv2.MedianBlur(objetivo, objetivoMod, 3);

            //ImageCard resultado = CalcularTemplateMatching(capturaMod, objetivoMod);            
            ImageCard rORB = SinEmgu.Calcular(capturaMod, objetivoMod, DetectorType.ORB);
            ImageCard rAKAZE = SinEmgu.Calcular(capturaMod, objetivoMod, DetectorType.AKAZE);
            ImageCard rSIFT = SinEmgu.Calcular(capturaMod, objetivoMod, DetectorType.SIFT, MatcherType.FLANN);
            Dispatcher.Invoke(() =>
            {
                //Cards.Add(resultado);
                Cards.Add(rORB);
                Cards.Add(rAKAZE);
                Cards.Add(rSIFT);
            });

        }

        

        //private static Mat? CapturarPantalla()
        //{
        //    var bounds = System.Windows.Forms.Screen.PrimaryScreen?.Bounds;

        //    if (bounds == null) return null;

        //    Bitmap bmp = new(bounds.Value.Width, bounds.Value.Height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
        //    using (Graphics g = Graphics.FromImage(bmp))
        //    {
        //        g.CopyFromScreen(0, 0, 0, 0, bmp.Size);
        //    }

        //    return bmp.ToMat();
        //}

    }
}