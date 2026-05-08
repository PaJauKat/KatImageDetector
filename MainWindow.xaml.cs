using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace KatImageDetector
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private async void BtnOnOff_Click(object sender, RoutedEventArgs e)
        {
            btnIniciar.IsEnabled = false;
            btnIniciar.Content = "Capturando...";

            await EscanearPantalla();

            btnIniciar.IsEnabled = true;
            btnIniciar.Content = "Capturar";
        }

        private async Task EscanearPantalla()
        {
            var pathImgObjetivo = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "boton_objetivo.png");
            Debug.WriteLine("Buscando imagen objetivo en: " + pathImgObjetivo);
            using Image<Bgr, byte> objetivo = new(pathImgObjetivo);
            using Mat? captura = CapturarPantalla();
            if (captura == null)
            {
                Debug.WriteLine("Captura fallida");
                return;
            }

            using Mat result = new();

            CvInvoke.MatchTemplate(captura, objetivo, result, TemplateMatchingType.CcoeffNormed);


            double maxValue = 0, minValue = 0;
            System.Drawing.Point maxLocation = new(), minLoc = new();

            CvInvoke.MinMaxLoc(result, ref minValue, ref maxValue, ref minLoc, ref maxLocation);

            Debug.WriteLine("MaxValue=" + maxValue);

            var rect = new System.Drawing.Rectangle(maxLocation, objetivo.Size);
            CvInvoke.Rectangle(captura, rect, new MCvScalar(0, 0, 255), 2);

            var rutaOutput = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                "KatLol",
                DateTime.Now.ToString("yyyyMMdd_HHmmss") + "_resultado.png"
                );
            captura.Save(rutaOutput);
            Debug.WriteLine("Resultado guardado en:" + rutaOutput);

            //Mostrar en UI
            BitmapSource result2UI = Mat2BitmapSource(captura);

            Dispatcher.Invoke(() =>
            {
                imgResultado.Source = result2UI;
                coincidenceResult.Text = "Result: " + maxValue;
            });

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
            return BitmapSource.Create(
                mat.Width,
                mat.Height,
                96,
                96,
                System.Windows.Media.PixelFormats.Bgr24,
                null,
                mat.DataPointer,
                mat.Step * mat.Rows,
                mat.Step);
        }
    }
}