#include <iostream> // TODO
#include <robot_design/render.h>
#include <stdexcept>
#include <string>
#include <tinyxml2.h>

namespace robot_design {

using tinyxml2::XMLAttribute;
using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLError;
using tinyxml2::XML_SUCCESS;

struct BitmapFontXMLVisitor : tinyxml2::XMLVisitor {
  BitmapFontXMLVisitor(BitmapFont &font) : font_(font) {}
  bool VisitEnter(const XMLElement &element, const XMLAttribute *attr) override;
  bool VisitExit(const XMLElement &element) override;

  BitmapFont &font_;
};

bool BitmapFontXMLVisitor::VisitEnter(const XMLElement &element,
                                      const XMLAttribute *attr) {
  const std::string element_name(element.Name());
  if (element_name == "page") {
    unsigned int id = 0;
    std::string file;
    while (attr) {
      const std::string attr_name(attr->Name());
      if (attr_name == "id") {
        id = attr->UnsignedValue();
      } else if(attr_name == "file") {
        file = attr->Value();
      }
      attr = attr->Next();
    }
    if (id >= font_.page_textures_.size()) {
      font_.page_textures_.resize(id + 1);
    }
    font_.page_textures_[id] = loadTexture(file);
  } else if (element_name == "char") {
    font_.chars_.emplace_back();
    BitmapFontChar &font_char = font_.chars_.back();
    while (attr) {
      const std::string attr_name(attr->Name());
      if (attr_name == "char") {
        font_char.char_ = attr->Value()[0];
      } else if (attr_name == "width") {
        font_char.width_ = attr->UnsignedValue();
      } else if (attr_name == "height") {
        font_char.height_ = attr->UnsignedValue();
      } else if (attr_name == "xoffset") {
        font_char.xoffset_ = attr->IntValue();
      } else if (attr_name == "yoffset") {
        font_char.yoffset_ = attr->IntValue();
      } else if (attr_name == "xadvance") {
        font_char.xadvance_ = attr->IntValue();
      } else if (attr_name == "x") {
        font_char.x_ = attr->IntValue();
      } else if (attr_name == "y") {
        font_char.y_ = attr->IntValue();
      } else if (attr_name == "page") {
        font_char.page_ = attr->UnsignedValue();
      }
      attr = attr->Next();
    }
  }
  return true;
}

bool BitmapFontXMLVisitor::VisitExit(const XMLElement &element) {
  return true;
}

BitmapFont::BitmapFont(const std::string &path) {
  XMLDocument document;
  XMLError error = document.LoadFile(path.c_str());
  if (error != XML_SUCCESS) {
    throw std::runtime_error(std::string("Could not load font XML: ") +
                             document.ErrorStr());
  }

  BitmapFontXMLVisitor visitor(*this);
  document.Accept(&visitor);
}

} // namespace robot_design
