
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
                TextoInfo = $"GoodMatches: {goodMatches.Count}"
            });

            return new MainWindow.ImageCard { 
                ImagenesInfo = infos,
                Title = "SIFT",
                Type = DetectorType.SIFT
            };
        }

        internal static KatImageDetector.MainWindow.ImageCard Calcular(Mat captura, Mat objetivo, DetectorType detectorType, MatcherType matcherType = MatcherType.BRUTE_FORCE)
        {
            var infos = new Collection<MainWindow.ImageInfo>();


            using Feature2D detector = detectorType switch
            {
                DetectorType.ORB => OpenCvSharp.ORB.Create(),
                DetectorType.AKAZE => OpenCvSharp.AKAZE.Create(),
                DetectorType.OpenCV_TemplateMatching => throw new NotImplementedException(),
                DetectorType.YOLO => throw new NotImplementedException(),
                DetectorType.SIFT => OpenCvSharp.Features2D.SIFT.Create(),
                _ => throw new NotSupportedException($"Detector type {detectorType} is not supported.")
            };

            using var capturaDescriptors = new Mat();
            detector.DetectAndCompute(captura, null, out var keypointsCaptura, capturaDescriptors);
            using var objetivoDescriptors = new Mat();
            detector.DetectAndCompute(objetivo, null, out var keypointsObjetivo, objetivoDescriptors);

            using DescriptorMatcher matcher = matcherType switch
            {
                MatcherType.BRUTE_FORCE => new OpenCvSharp.BFMatcher(NormTypes.Hamming, crossCheck: false),
                MatcherType.FLANN => detectorType == DetectorType.SIFT ? 
                    new OpenCvSharp.FlannBasedMatcher(new KDTreeIndexParams(5), new SearchParams(50)) : throw new NotSupportedException($"FLANN matcher is only supported for SIFT."),
                _ => throw new NotSupportedException($"Matcher type {matcherType} is not supported.")
            };

            DMatch[][] matches = matcher.KnnMatch(objetivoDescriptors, capturaDescriptors, 2);

            var goodMatches = new List<DMatch>();
            foreach (var m in matches)
            {
                if (m.Length == 2 && m[0].Distance < 0.75 * m[1].Distance)
                {
                    goodMatches.Add(m[0]);
                }
            }

            // Agregando la imagen con los matches dibujados
            using var result = new Mat();
            Cv2.DrawMatches(
                objetivo, keypointsObjetivo,
                captura, keypointsCaptura,
                goodMatches,
                result);


            infos.Add(new MainWindow.ImageInfo
            {
                Image = result.ToBitmapSource(),
                TextoInfo = $"GoodMatches: {goodMatches.Count}",
                Label = "Matches"
            });

            // Agregando la homografia si es posible
            if (goodMatches.Count >= 4)
            {
                var srcPts = OpenCvSharp.InputArray.Create(goodMatches.Select(x => keypointsObjetivo[x.QueryIdx].Pt).ToArray());
                var dstPts = OpenCvSharp.InputArray.Create(goodMatches.Select(x => keypointsCaptura[x.TrainIdx].Pt).ToArray());

                using var mask = new Mat();

                using Mat homography = Cv2.FindHomography(
                    srcPts,
                    dstPts,
                    method: HomographyMethods.Ransac,
                    mask: mask
                    );

                if (homography != null)
                {
                    using var warped = new Mat();
                    Cv2.WarpPerspective(objetivo, warped, homography, new OpenCvSharp.Size(captura.Width, captura.Height));

                    infos.Add(new MainWindow.ImageInfo
                    {
                        Image = warped.ToBitmapSource(),
                        TextoInfo = "Homography calculated successfully.",
                        Label = "Warped"
                    });

                    Point2f[] objCorners =
                    [
                        new(0,0),
                        new(objetivo.Width, 0),
                        new(objetivo.Width, objetivo.Height),
                        new(0, objetivo.Height)
                    ];

                    // Projectando los puntos con la homografia en la captura
                    Point2f[] capturaCorners = Cv2.PerspectiveTransform(objCorners, homography);
                    using var capturaCopy = captura.Clone();
                    Cv2.Polylines(capturaCopy, [capturaCorners.Select(x => x.ToPoint()).ToArray()], true, Scalar.Green);

                    infos.Add(new MainWindow.ImageInfo
                    {
                        Image = capturaCopy.ToBitmapSource(),
                        TextoInfo = $"Inliers: {Cv2.CountNonZero(mask)}, Ratio: {(double) Cv2.CountNonZero(mask)/goodMatches.Count}",
                        Label = "Homography"
                    });
                }
            }
            

            return new MainWindow.ImageCard
            {
                ImagenesInfo = infos,
                Title = detectorType.ToString(),
                Type = detectorType
            };
        }
    }
}
