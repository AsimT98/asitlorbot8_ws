#include "asitlorbot8_firmware/asitlorbot8_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <sstream>
#include <iomanip>

namespace asitlorbot8_firmware
{
    asitlorbotInterface::asitlorbotInterface()
    {
    }

    asitlorbotInterface::~asitlorbotInterface()
    {
        if (arduino_.IsOpen())
        {
            try
            {
                arduino_.Close();
            }
            catch (...)
            {
                RCLCPP_FATAL_STREAM(rclcpp::get_logger("asitlorbotInterface"), "Something went wrong while closing connection with port " << port_);
            }
        }
    }

    CallbackReturn asitlorbotInterface::on_init(const hardware_interface::HardwareInfo &hardware_info)
    {
        CallbackReturn result = hardware_interface::SystemInterface::on_init(hardware_info);
        if (result != CallbackReturn::SUCCESS)
        {
            return result;
        }

        try
        {
            port_ = info_.hardware_parameters.at("port");
        }
        catch (const std::out_of_range &e)
        {
            RCLCPP_FATAL(rclcpp::get_logger("asitlorbotInterface"), "No Serial Port provided! Aborting");
            return CallbackReturn::FAILURE;
        }

        velocity_commands_.reserve(info_.joints.size());
        position_states_.reserve(info_.joints.size());
        velocity_states_.reserve(info_.joints.size());
        position_offsets_.reserve(info_.joints.size());

        return CallbackReturn::SUCCESS;
    }

    std::vector<hardware_interface::StateInterface> asitlorbotInterface::export_state_interfaces()
    {
        std::vector<hardware_interface::StateInterface> state_interfaces;

        for (size_t i = 0; i < info_.joints.size(); i++)
        {
            state_interfaces.emplace_back(hardware_interface::StateInterface(
                info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_states_[i]));
            state_interfaces.emplace_back(hardware_interface::StateInterface(
                info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocity_states_[i]));
        }

        return state_interfaces;
    }

    std::vector<hardware_interface::CommandInterface> asitlorbotInterface::export_command_interfaces()
    {
        std::vector<hardware_interface::CommandInterface> command_interfaces;

        for (size_t i = 0; i < info_.joints.size(); i++)
        {
            command_interfaces.emplace_back(hardware_interface::CommandInterface(
                info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocity_commands_[i]));
        }

        return command_interfaces;
    }

    CallbackReturn asitlorbotInterface::on_activate(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(rclcpp::get_logger("asitlorbotInterface"), "Starting robot hardware ...");

        // Reset commands and states
        velocity_commands_ = {0.0, 0.0, 0.0, 0.0};
        position_states_ = {0.0, 0.0, 0.0, 0.0};
        velocity_states_ = {0.0, 0.0, 0.0, 0.0};
        position_offsets_ = {0.0, 0.0, 0.0, 0.0};

        try
        {
            arduino_.Open(port_);
            arduino_.SetBaudRate(LibSerial::BaudRate::BAUD_115200);
        }
        catch (...)
        {
            RCLCPP_FATAL_STREAM(rclcpp::get_logger("asitlorbotInterface"),
                                "Something went wrong while interacting with port " << port_);
            return CallbackReturn::FAILURE;
        }

        RCLCPP_INFO(rclcpp::get_logger("asitlorbotInterface"),
                    "Hardware started, ready to take commands");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn asitlorbotInterface::on_deactivate(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(rclcpp::get_logger("asitlorbotInterface"), "Stopping robot hardware ...");

        if (arduino_.IsOpen())
        {
            try
            {
                arduino_.Close();
            }
            catch (...)
            {
                RCLCPP_FATAL_STREAM(rclcpp::get_logger("asitlorbotInterface"),
                                    "Something went wrong while closing connection with port " << port_);
            }
        }

        RCLCPP_INFO(rclcpp::get_logger("asitlorbotInterface"), "Hardware stopped");
        return CallbackReturn::SUCCESS;
    }

    hardware_interface::return_type asitlorbotInterface::read(const rclcpp::Time &, const rclcpp::Duration &period)
    {
        if (arduino_.IsDataAvailable())
        {
            std::string message;
            arduino_.ReadLine(message);
            std::stringstream ss(message);
            std::string res;
            int multiplier = 1;
            while (std::getline(ss, res, ','))
            {
                multiplier = res.at(1) == 'p' ? 1 : -1;

                if (res.at(0) == 'r')
                {
                    velocity_states_.at(0) = multiplier * std::stod(res.substr(2, res.size()));
                    position_states_.at(0) += velocity_states_.at(0) * period.seconds();
                }
                else if (res.at(0) == 'l')
                {
                    velocity_states_.at(1) = multiplier * std::stod(res.substr(2, res.size()));
                    position_states_.at(1) += velocity_states_.at(1) * period.seconds();
                }
            }
            RCLCPP_INFO_STREAM(rclcpp::get_logger("asitlorbotInterface"), "read a message: " << message);
        }
        return hardware_interface::return_type::OK;
    }

    hardware_interface::return_type asitlorbotInterface::write(const rclcpp::Time &, const rclcpp::Duration &)
    {
        std::stringstream message_stream;
        char right_wheel_sign = velocity_commands_.at(0) >= 0 ? 'p' : 'n';
        char left_wheel_sign = velocity_commands_.at(1) >= 0 ? 'p' : 'n';
        std::string compensate_zeros_right = std::abs(velocity_commands_.at(0)) < 10.0 ? "0" : "";
        std::string compensate_zeros_left = std::abs(velocity_commands_.at(1)) < 10.0 ? "0" : "";

        message_stream << std::fixed << std::setprecision(2) 
                       << "r" << right_wheel_sign << compensate_zeros_right << std::abs(velocity_commands_.at(0))
                       << ",l" << left_wheel_sign << compensate_zeros_left << std::abs(velocity_commands_.at(1)) 
                       << ",";

        try
        {
            arduino_.Write(message_stream.str());
        }
        catch (...)
        {
            RCLCPP_ERROR_STREAM(rclcpp::get_logger("asitlorbotInterface"), "Something went wrong while sending the message " << message_stream.str() << " to the port " << port_);
            return hardware_interface::return_type::ERROR;
        }

        return hardware_interface::return_type::OK;
    }

}  // namespace asitlorbot8_firmware

PLUGINLIB_EXPORT_CLASS(asitlorbot8_firmware::asitlorbotInterface, hardware_interface::SystemInterface)
