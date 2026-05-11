using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace KatImageDetector
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        public class ImageCard
        {
            public string Title { get; set; }
            public DetectorType Type { get; set; }
            public Collection<ImageInfo> ImagenesInfo { get; set; } = [];
        }

        public class ImageInfo
        {
            public BitmapSource Image { get; set; }
            public string Resultado { get; set; }
        }

        public ObservableCollection<ImageCard> Cards { get; set; } = new();
        public Popup _popup;

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

        private async void ImgThumb_MouseEnter(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (sender is System.Windows.Controls.Image image)
            {
                Grid grid = (Grid)VisualTreeHelper.GetParent(image);
                if (grid == null)
                {
                    return;
                }

                Popup popup = (Popup)VisualTreeHelper.GetChild(grid, 1);
                if (popup == null)
                {
                    return;
                }
                popup.IsOpen = true;
            }
        }

        private async void ImgThumb_MouseLeave(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (sender is System.Windows.Controls.Image image)
            {
                Grid grid = (Grid)VisualTreeHelper.GetParent(image);
                if (grid == null)
                {
                    return;
                }

                Popup popup = (Popup)VisualTreeHelper.GetChild(grid, 1);
                if (popup == null)
                {
                    return;
                }
                popup.IsOpen = false;
            }
        }

        private async Task EscanearPantalla()
        {

            var pathImgObjetivo = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "boton_objetivo.png");
            Debug.WriteLine("Buscando imagen objetivo en: " + pathImgObjetivo);
            using Mat objetivo = CvInvoke.Imread(pathImgObjetivo, ImreadModes.Grayscale);
            using var objetivoMod = new Mat();
            CvInvoke.MedianBlur(objetivo,objetivoMod, 3);

            using Mat? captura = CapturarPantalla();
            if (captura == null)
            {
                Debug.WriteLine("Captura fallida");
                return;
            }

            using var capturaMod = new Mat();
            CvInvoke.CvtColor(captura, capturaMod, ColorConversion.Bgr2Gray);

            ImageCard resultado = CalcularTemplateMatching(capturaMod, objetivoMod);            
            ImageCard rORB = CalcularORB(capturaMod, objetivoMod);
            using var capturaMod2 = new Mat(capturaMod, new Rectangle(0, 0, capturaMod.Width / 2, capturaMod.Height));
            ImageCard rAKAZE = CalcularAKAZE(capturaMod2, objetivoMod);
            Dispatcher.Invoke(() =>
            {
                Cards.Add(resultado);
                Cards.Add(rORB);
                Cards.Add(rAKAZE);
            });

        }

        private static ImageCard CalcularAKAZE(Mat capturaOriginal, Mat referencia)
        {
            var akaze = new Emgu.CV.Features2D.AKAZE();

            //Imagen de referencia
            var referenceKp = new VectorOfKeyPoint();
            using var refDescriptors = new Mat();
            akaze.DetectAndCompute(referencia, null, referenceKp, refDescriptors, false);

            //Imagen de captura
            var capturaKp = new VectorOfKeyPoint();
            using var capturaDescriptors = new Mat();
            akaze.DetectAndCompute(capturaOriginal, null, capturaKp, capturaDescriptors, false);

            //AKAZE usa descriptores binarios, por lo que se puede usar Hamming
            BFMatcher matcher = new(Emgu.CV.Features2D.DistanceType.Hamming);
            var matches = new VectorOfVectorOfDMatch();
            matcher.KnnMatch(refDescriptors, capturaDescriptors, matches, k: 2);
            Debug.WriteLine($"AKAZE - Matches encontrados: {matches.Size}");

            // 4. Filtrado (Lowe's Ratio Test)
            // Solo nos quedamos con matches donde la distancia sea significativamente menor que la segunda mejor
            var goodMatches = new VectorOfDMatch();
            for (int i = 0; i < matches.Size; i++)
            {
                using var matchRow = matches[i];
                if (matchRow.Size >= 2)
                {
                    var matchArray = matchRow.ToArray();
                    if (matchArray[0].Distance < 0.8 * matchArray[1].Distance)
                    {
                        goodMatches.Push(new MDMatch[] { matchArray[0] });
                    }
                }
            }

            //Imprimiendo
            Debug.WriteLine($"AKAZE - Keypoints referencia: {referenceKp.Size}, Keypoints captura: {capturaKp.Size}");

            //Homografia
            if (goodMatches.Size >= 4)
            {
                using Mat homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(
                    referenceKp, capturaKp, matches, null, 5
                    );

                if (homography != null)
                {
                    //Limites del objeto
                    var rect = new Rectangle(System.Drawing.Point.Empty, referencia.Size);

                    PointF[] objCorners = new PointF[]
                    {
                        new PointF(rect.Left, rect.Top),
                        new PointF(rect.Right, rect.Top),
                        new PointF(rect.Right, rect.Bottom),
                        new PointF(rect.Left, rect.Bottom)
                    };

                    objCorners = CvInvoke.PerspectiveTransform(objCorners, homography);

                    for (int i = 0; i < 4; i++)
                    {
                        CvInvoke.Line(
                            capturaOriginal, 
                            System.Drawing.Point.Round(objCorners[i]),
                            System.Drawing.Point.Round(objCorners[(i + 1) % 4]), 
                            new MCvScalar(0, 255, 0), 
                            2);
                    }
                }
                else
                {
                    Debug.WriteLine("Homografía no encontrada para AKAZE");
                }
            }
            else
            {
                Debug.WriteLine($"AKAZE - No se encontraron suficientes matches buenos para homografía (encontrados: {goodMatches.Size})");
            }

            /*
            using var result = new Mat();
            try
            {
                Emgu.CV.Features2D.Features2DToolbox.DrawMatches(
                    referencia, referenceKp,
                    capturaOriginal, capturaKp,
                    goodMatches,
                    result,
                    new MCvScalar(0, 255, 0),
                    new MCvScalar(255, 0, 0),
                    null,
                    Emgu.CV.Features2D.Features2DToolbox.KeypointDrawType.Default
                );
            }
            catch (Exception ex)
            {
                Debug.WriteLine("Error dibujando matches para AKAZE: " + ex.Message);
            }
            */

            return new ImageCard
            {
                Title = "AKAZE",
                Image = Mat2BitmapSource(capturaOriginal),
                Resultado = "Matches buenos: " + goodMatches.Length,
                Type = DetectorType.AKAZE
            };


        }

        private static ImageCard CalcularORB(Mat capturaOriginal, Mat referencia)
        {
            var orb = new Emgu.CV.Features2D.ORB(500);

            //Imagen referencia
            VectorOfKeyPoint kp1 = new();
            Mat desc1 = new();
            orb.DetectAndCompute(referencia, null, kp1, desc1, false);

            //Imagen captura
            VectorOfKeyPoint kp2 = new();
            Mat desc2 = new();
            orb.DetectAndCompute(capturaOriginal, null, kp2, desc2, false);

            Debug.WriteLine($"kp1: {kp1.Size}, kp2: {kp2.Size}");
            Debug.WriteLine($"desc1 empty: {desc1.IsEmpty}, desc2 empty: {desc2.IsEmpty}");


            if (desc1 == null || desc1.IsEmpty || desc2 == null || desc2.IsEmpty)
            {
                Debug.WriteLine("Descriptors vacíos → ORB no encontró features");
                return new ImageCard { Title = "ORB", Resultado = "Sin features", Type = DetectorType.ORB };
            }

            //Matching
            var bf = new Emgu.CV.Features2D.BFMatcher(Emgu.CV.Features2D.DistanceType.Hamming);
            VectorOfVectorOfDMatch matches = new();
            bf.KnnMatch(desc1,desc2, matches, k:2);


            if (matches.Size == 0)
            {
                Debug.WriteLine("No matches returned by KnnMatch");
                return new ImageCard { Title = "ORB", Resultado = "No matches", Type = DetectorType.ORB };
            }

            //Aplicar ratio test de Lowe
            VectorOfDMatch goodMatches = new();
            for (int i = 0; i < matches.Size; i++)
            {
                // matches[i] devuelve un VectorOfDMatch (los k:2 vecinos)
                using (var row = matches[i])
                {
                    if (row.Size >= 2)
                    {
                        var matchArray = row.ToArray(); // [0] es el mejor, [1] es el segundo mejor

                        // Aplicar ratio test de Lowe
                        if (matchArray[0].Distance < 0.8 * matchArray[1].Distance)
                        {
                            // Empujamos solo el mejor match (el índice 0)
                            goodMatches.Push(new MDMatch[] { matchArray[0] });
                        }
                    }
                }
            }

            //CvInvoke.FindHomography(kp1, kp2, goodMatches, Emgu.CV.Features2D.Homogra, 5, out Mat homography);
            using var result = new Mat();
            try
            {

                Emgu.CV.Features2D.Features2DToolbox.DrawMatches(
                    referencia, kp1,
                    capturaOriginal, kp2,
                    goodMatches,
                    result,
                    new MCvScalar(0, 255, 0),
                    new MCvScalar(255, 0, 0)
                    );

            }
            catch (Exception ex)
            {
                Debug.WriteLine("Error dibujando matches: " + ex.Message);
            }

            return new ImageCard
            {
                Title = "ORB",
                Image = Mat2BitmapSource(result),
                Resultado = "Matches buenos: " + goodMatches.Length,
                Type = DetectorType.ORB
            };

        }

        private static ImageCard CalcularTemplateMatching(Mat capturaOriginal, Mat objetivo)
        {
            using Mat result = new();
            using Mat captura = capturaOriginal.Clone();

            CvInvoke.MatchTemplate(captura, objetivo, result, TemplateMatchingType.CcoeffNormed);

            double maxValue = 0, minValue = 0;
            System.Drawing.Point maxLocation = new(), minLoc = new();

            CvInvoke.MinMaxLoc(result, ref minValue, ref maxValue, ref minLoc, ref maxLocation);

            var rect = new System.Drawing.Rectangle(maxLocation, objetivo.Size);
            CvInvoke.Rectangle(captura, rect, new MCvScalar(0, 0, 255), 2);

            var rutaOutput = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                "KatLol",
                DateTime.Now.ToString("yyyyMMdd_HHmmss") + "_resultado.png"
                );
            captura.Save(rutaOutput);
            Debug.WriteLine("Resultado guardado en:" + rutaOutput);
            var bitmapSource = Mat2BitmapSource(captura);

            return new ImageCard
            {
                Title = "Simple MatchTemplate",
                Image = bitmapSource,
                Resultado = "Coincidencia: " + maxValue.ToString("F4"),
                Type = DetectorType.OpenCV_TemplateMatching
            };
        }

        private static Mat? CapturarPantalla()
        {
            var bounds = System.Windows.Forms.Screen.PrimaryScreen?.Bounds;

            if (bounds == null) return null;

            Bitmap bmp = new(bounds.Value.Width, bounds.Value.Height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.CopyFromScreen(0, 0, 0, 0, bmp.Size);
            }

            return bmp.ToMat();
        }


        public static BitmapSource Mat2BitmapSource(Mat mat)
        {
            if (mat.IsEmpty) return null;

            System.Windows.Media.PixelFormat format;

            // 1. Determinar el formato según los canales
            if (mat.NumberOfChannels == 1)
            {
                format = System.Windows.Media.PixelFormats.Gray8;
            }
            else if (mat.NumberOfChannels == 3)
            {
                format = System.Windows.Media.PixelFormats.Bgr24;
            }
            else if (mat.NumberOfChannels == 4)
            {
                format = System.Windows.Media.PixelFormats.Bgra32;
            }
            else
            {
                throw new ArgumentException($"Formato de Mat no soportado: {mat.NumberOfChannels} canales.");
            }

            // 2. Calcular el tamaño total del buffer de pixeles
            // Step es el número de bytes por fila (incluyendo el padding de memoria de OpenCV)
            int stride = mat.Step;
            int bufferSize = stride * mat.Rows;

            return BitmapSource.Create(
                mat.Width,
                mat.Height,
                96,
                96,
                format,
                null,
                mat.DataPointer,
                bufferSize,
                stride);
        }

        private void Image_MouseLeave(object sender, System.Windows.Input.MouseEventArgs e)
        {

        }
    }
}