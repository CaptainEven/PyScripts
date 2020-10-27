
void WritePlyText(const std::string &path,
                  const std::vector<FusedPoint> &points)
{
  std::ofstream file(path);
  assert(file.is_open());

  size_t valid_cnt = 0;
  for (const auto &point : points)
  {
    if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z) || std::isnan(point.nx) || std::isnan(point.ny) || std::isnan(point.nz))
    {
      //std::cout << "[Nan]: " << point.x << " " << point.y << " "
      // << point.z << " " << point.nx << " " << point.ny << " "
      // << point.nz << std::endl;
      continue;
    }

    valid_cnt += 1;
  }
  std::cout << "total " << valid_cnt << " valid points." << std::endl;

  file << "ply" << std::endl;
  file << "format ascii 1.0" << std::endl;
  //file << "element vertex " << points.size() << std::endl;
  file << "element vertex " << valid_cnt << std::endl;
  file << "property float x" << std::endl;
  file << "property float y" << std::endl;
  file << "property float z" << std::endl;
  file << "property float nx" << std::endl;
  file << "property float ny" << std::endl;
  file << "property float nz" << std::endl;
  file << "property uchar red" << std::endl;
  file << "property uchar green" << std::endl;
  file << "property uchar blue" << std::endl;
  file << "end_header" << std::endl;

  for (const auto &point : points)
  {
    if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z) || std::isnan(point.nx) || std::isnan(point.ny) || std::isnan(point.nz))
    {
      //std::cout << "Nan: " << point.x << " " << point.y << " "
      // << point.z << " " << point.nx << " " << point.ny
      // << point.nz << std::endl;
      continue;
    }

    file << point.x << " " << point.y << " " << point.z << " " << point.nx
         << " " << point.ny << " " << point.nz << " "
         << static_cast<int>(point.r) << " " << static_cast<int>(point.g) << " "
         << static_cast<int>(point.b) << std::endl;
  }

  file.close();
}