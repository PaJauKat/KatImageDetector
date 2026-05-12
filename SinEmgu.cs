
using OpenCvSharp;
using OpenCvSharp.Flann;
using OpenCvSharp.WpfExtensions;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KatImageDetector
{
    internal class SinEmgu
    {
        internal static KatImageDetector.MainWindow.ImageCard CalcularSIFT()
        {
            var infos = new Collection<MainWindow.ImageInfo>();

            using var captura = new Mat(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "pantallazo.png"), ImreadModes.Grayscale);
            using var objetivo = new Mat(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "boton_objetivo.png"), ImreadModes.Grayscale);

            using var sift = OpenCvSharp.Features2D.SIFT.Create();

            using var capturaDescriptors = new Mat();
            sift.DetectAndCompute(captura, null, out var keypointsCaptura, capturaDescriptors);
            using var objetivoDescriptors = new Mat();
            sift.DetectAndCompute(objetivo, null, out var keypointsObjetivo, objetivoDescriptors);


            using var matcher = new FlannBasedMatcher(new KDTreeIndexParams(5), new SearchParams(50));
            DMatch[][] matches = matcher.KnnMatch(objetivoDescriptors, capturaDescriptors, 2);

            var goodMatches = new List<DMatch>();
            foreach (var m in matches)
            {
                if(m.Length == 2 && m[0].Distance < 0.75 * m[1].Distance)
                {
                    goodMatches.Add(m[0]);
                }
            }

            using var result = new Mat();
            Cv2.DrawMatches(
                objetivo, keypointsObjetivo,
                captura, keypointsCaptura,
                goodMatches,
                result);


          
            infos.Add(new MainWindow.ImageInfo { 
                Image = result.ToBitmapSource(),
                Resultado = $"GoodMatches: {goodMatches.Count}"
            });

            return new MainWindow.ImageCard { 
                ImagenesInfo = infos,
                Title = "SIFT",
                Type = DetectorType.SIFT
            };
        }

        internal static KatImageDetector.MainWindow.ImageCard CalcularAKAZE()
        {
            var infos = new Collection<MainWindow.ImageInfo>();

            using var captura = new Mat(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "pantallazo.png"), ImreadModes.Grayscale);
            using var objetivo = new Mat(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "boton_objetivo.png"), ImreadModes.Grayscale);

            using var sift = OpenCvSharp.Features2D.SIFT.Create();

            using var capturaDescriptors = new Mat();
            sift.DetectAndCompute(captura, null, out var keypointsCaptura, capturaDescriptors);
            using var objetivoDescriptors = new Mat();
            sift.DetectAndCompute(objetivo, null, out var keypointsObjetivo, objetivoDescriptors);


            using var matcher = new FlannBasedMatcher(new KDTreeIndexParams(5), new SearchParams(50));
            DMatch[][] matches = matcher.KnnMatch(objetivoDescriptors, capturaDescriptors, 2);

            var goodMatches = new List<DMatch>();
            foreach (var m in matches)
            {
                if (m.Length == 2 && m[0].Distance < 0.75 * m[1].Distance)
                {
                    goodMatches.Add(m[0]);
                }
            }

            using var result = new Mat();
            Cv2.DrawMatches(
                objetivo, keypointsObjetivo,
                captura, keypointsCaptura,
                goodMatches,
                result);



            infos.Add(new MainWindow.ImageInfo
            {
                Image = result.ToBitmapSource(),
                Resultado = $"GoodMatches: {goodMatches.Count}"
            });

            return new MainWindow.ImageCard
            {
                ImagenesInfo = infos,
                Title = "SIFT",
                Type = DetectorType.SIFT
            };
        }
    }
}
