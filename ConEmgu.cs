using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using static KatImageDetector.MainWindow;

namespace KatImageDetector
{
    internal class ConEmgu
    {

        private static ImageCard CalcularSIFT(Mat capturaOriginal, Mat referencia)
        {
            Debug.WriteLine("------------------------------");
            var infos = new Collection<ImageInfo>();
            var sift = new Emgu.CV.Features2D.SIFT();

            //Imagen de referencia
            var referenceKp = new VectorOfKeyPoint();
            using var refDescriptors = new Mat();
            sift.DetectAndCompute(referencia, null, referenceKp, refDescriptors, false);

            //Imagen de captura
            var capturaKp = new VectorOfKeyPoint();
            using var capturaDescriptors = new Mat();
            sift.DetectAndCompute(capturaOriginal, null, capturaKp, capturaDescriptors, false);

            //Matching con FLANN
            var matcher = new Emgu.CV.Features2D.FlannBasedMatcher(new Emgu.CV.Flann.KdTreeIndexParams(5), new Emgu.CV.Flann.SearchParams(50));
            var matches = new VectorOfVectorOfDMatch();
            matcher.KnnMatch(refDescriptors, capturaDescriptors, matches, k: 2);
            Debug.WriteLine($"SIFT - Matches encontrados: {matches.Size}");

            //Aplicar ratio test de Lowe
            var goodMatches = new VectorOfDMatch();
            var tempMatches = new List<MDMatch>();

            for (int i = 0; i < matches.Size; i++)
            {
                var matchRow = matches[i];
                if (matchRow.Size >= 2)
                {
                    if (matchRow[0].Distance < 0.7 * matchRow[1].Distance)
                    {
                        //goodMatches.Push(new MDMatch[] { matchArray[0] });
                        tempMatches.Add(matchRow[0]);
                    }
                }
            }

            goodMatches.Push(tempMatches.ToArray());

            // DrawMatches para visualizar los buenos matches
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

                infos.Add(new ImageInfo
                {
                    Image = Mat2BitmapSource(result),
                    Resultado = "Matches buenos: " + goodMatches.Size
                });

            }
            catch (Exception ex)
            {
                Debug.WriteLine("Error dibujando matches para SIFT: " + ex.Message);
            }


            //Homografia
            using var capturaCopy = capturaOriginal.Clone();
            if (goodMatches.Size >= 4)
            {
                Debug.WriteLine($"SIFT - Calculando homografía con {goodMatches.Size} buenos matches...");
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
                            capturaCopy,
                            System.Drawing.Point.Round(objCorners[i]),
                            System.Drawing.Point.Round(objCorners[(i + 1) % 4]),
                            new MCvScalar(0, 255, 0),
                            2);
                    }

                    infos.Add(new ImageInfo
                    {
                        Image = Mat2BitmapSource(capturaCopy),
                        Resultado = "Matches buenos: " + goodMatches.Size
                    });
                }
            }
            else
            {
                Debug.WriteLine($"SIFT - No se encontraron suficientes matches buenos para homografía (encontrados: {goodMatches.Size})");
            }

            return new ImageCard
            {
                Title = "SIFT",
                ImagenesInfo = infos,
                Type = DetectorType.SIFT
            };
        }

        private static ImageCard CalcularAKAZE(Mat capturaOriginal, Mat referencia)
        {

            Debug.WriteLine("------------------------------");
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
                var matchRow = matches[i];
                if (matchRow.Size >= 2)
                {
                    var matchArray = matchRow.ToArray();
                    if (matchArray[0].Distance < 0.8 * matchArray[1].Distance)
                    {
                        goodMatches.Push(new MDMatch[] { matchArray[0] });
                    }
                }
            }


            using var capturaCopy = capturaOriginal.Clone();
            //Imprimiendo
            Debug.WriteLine($"AKAZE - Keypoints referencia: {referenceKp.Size}, Keypoints captura: {capturaKp.Size}");

            //Homografia
            if (goodMatches.Size >= 4)
            {
                Debug.WriteLine($"AKAZE - Calculando homografía con {goodMatches.Size} buenos matches...");
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
                            capturaCopy,
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


            var infos = new Collection<ImageInfo>();
            infos.Add(new ImageInfo
            {
                Image = Mat2BitmapSource(capturaCopy),
                Resultado = "Matches buenos: " + goodMatches.Size
            });

            infos.Add(new ImageInfo
            {
                Image = Mat2BitmapSource(result),
                Resultado = "Matches: " + matches.Size + ", Matches buenos: " + goodMatches.Size
            });



            return new ImageCard
            {
                Title = "AKAZE",
                ImagenesInfo = infos,
                Type = DetectorType.AKAZE
            };


        }

        private static ImageCard CalcularORB(Mat capturaOriginal, Mat referencia)
        {
            Debug.WriteLine("------------------------------");
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
                return new ImageCard { Title = "ORB", ImagenesInfo = [], Type = DetectorType.ORB };
            }

            //Matching
            var bf = new Emgu.CV.Features2D.BFMatcher(Emgu.CV.Features2D.DistanceType.Hamming);
            VectorOfVectorOfDMatch matches = new();
            bf.KnnMatch(desc1, desc2, matches, k: 2);


            if (matches.Size == 0)
            {
                Debug.WriteLine("No matches returned by KnnMatch");
                return new ImageCard { Title = "ORB", ImagenesInfo = [], Type = DetectorType.ORB };
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
                Type = DetectorType.ORB,
                ImagenesInfo =
                [
                    new ImageInfo
                    {
                        Image = Mat2BitmapSource(result),
                        Resultado = "Matches buenos: " + goodMatches.Size
                    }
                ]
            };
        }

        private static ImageCard CalcularTemplateMatching(Mat capturaOriginal, Mat objetivo)
        {
            Debug.WriteLine("------------------------------");
            using Mat result = new();
            using Mat captura = capturaOriginal.Clone();

            CvInvoke.MatchTemplate(captura, objetivo, result, TemplateMatchingType.CcoeffNormed);

            double maxValue = 0, minValue = 0;
            System.Drawing.Point maxLocation = new(), minLoc = new();

            CvInvoke.MinMaxLoc(result, ref minValue, ref maxValue, ref minLoc, ref maxLocation);

            var rect = new System.Drawing.Rectangle(maxLocation, objetivo.Size);
            CvInvoke.Rectangle(captura, rect, new MCvScalar(0, 0, 255), 2);

            //var rutaOutput = Path.Combine(
            //    Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
            //    "KatLol",
            //    DateTime.Now.ToString("yyyyMMdd_HHmmss") + "_resultado.png"
            //    );
            //captura.Save(rutaOutput);
            //Debug.WriteLine("Resultado guardado en:" + rutaOutput);
            var bitmapSource = Mat2BitmapSource(captura);

            return new ImageCard
            {
                Title = "Simple MatchTemplate",
                ImagenesInfo =
                [
                    new ImageInfo
                    {
                        Image = bitmapSource,
                        Resultado = "Coincidencia: " + maxValue.ToString("F4")
                    }
                ],
                Type = DetectorType.OpenCV_TemplateMatching
            };
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
    }
}
